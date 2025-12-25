import os
ITEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ITEM_DIR)
import sys
import time

import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
import numpy as np

DEVICE = 'cpu' # 
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import_path = os.path.join("D:/", "11_Code", "Small-Target-Motion-Detectors", "python")
dataset_pth = os.path.join(ITEM_DIR, 'maritime', 'videos')
sys.path.append(import_path)
from smalltargetmotiondetectors.api import (instancing_model, inference) # type: ignore
from utils import FrameIterator, nms




class FastVideoEvaluator:
    def __init__(self, video_preds_sparse, video_gts):
        """
        初始化并执行耗时的预计算匹配逻辑。
        :param video_preds_sparse: list of lists [[(y,x,score), ...], ...] (每一帧必须按 score 降序排列)
        :param video_gts: list of lists [[{'bbox':...}, ...], ...]
        """
        self.num_frames = len(video_preds_sparse)
        
        # --- 1. 数据转换 (Data Preparation) ---
        self.gt_boxes_np = [] 
        self.gt_ids = []
        self.total_gt_count = 0
        
        # 处理 GT
        for f_idx, gt_list in enumerate(video_gts):
            boxes = []
            ids = []
            for i, item in enumerate(gt_list):
                x, y, w, h = item['bbox']
                boxes.append([x, y, x+w, y+h])
                tid = item.get('track_id', i)
                ids.append(f"{f_idx}_{tid}") # Global Unique ID
            
            self.gt_boxes_np.append(np.array(boxes) if boxes else np.empty((0, 4)))
            self.gt_ids.append(ids)
            self.total_gt_count += len(boxes)

        # 处理 Predictions
        self.pred_points_np = []
        self.pred_scores_np = []
        self.global_preds = [] # 用于 AP 计算

        for f_idx, preds in enumerate(video_preds_sparse):
            if not preds:
                self.pred_points_np.append(np.empty((0, 2)))
                self.pred_scores_np.append(np.empty((0,)))
                continue
                
            # 输入数据: [(y, x, score), ...]
            data = np.array(preds) 
            points = data[:, :2]  # y, x
            scores = data[:, 2]   # score
            
            self.pred_points_np.append(points)
            self.pred_scores_np.append(scores)
            
            # 收集用于 AP 全局排序的数据
            n_p = len(scores)
            batch_global = np.column_stack((
                scores, 
                np.full(n_p, f_idx), 
                np.arange(n_p)
            ))
            self.global_preds.append(batch_global)

        # AP 预处理: 全局排序
        if self.global_preds:
            self.all_preds_sorted = np.vstack(self.global_preds)
            sort_idx = np.argsort(-self.all_preds_sorted[:, 0]) # 降序
            self.all_preds_sorted = self.all_preds_sorted[sort_idx]
        else:
            self.all_preds_sorted = np.empty((0, 3))

        # --- 2. 核心加速：预计算匹配矩阵 (Pre-compute Matches) ---
        # self.match_cache[f] 是一个 Boolean 矩阵 (N_pred, M_gt)
        self.match_cache = self._precompute_matches()

    def _precompute_matches(self):
        """利用广播机制一次性计算所有点与框的包含关系"""
        cache = []
        for f_idx in range(self.num_frames):
            points = self.pred_points_np[f_idx] # (N, 2)
            boxes = self.gt_boxes_np[f_idx]     # (M, 4)
            
            if len(points) == 0 or len(boxes) == 0:
                cache.append(None)
                continue
            
            # 向量化判断 Point in Box
            # points[:, 0] 是 y, boxes 顺序是 [x1, y1, x2, y2]
            in_y = (points[:, 0:1] >= boxes[:, 1:2].T) & (points[:, 0:1] < boxes[:, 3:4].T)
            in_x = (points[:, 1:2] >= boxes[:, 0:1].T) & (points[:, 1:2] < boxes[:, 2:3].T)
            
            cache.append(in_y & in_x)
        return cache


    #  功能函数 1: 计算 AR @ Top K (例如 AR@10)
    def calculate_ar_top_k(self, k=10):
        """
        计算 AR@K (Top K Response per frame)
        前提：输入数据已经按 Score 降序排列 (Top-N 截取策略已保证这点)
        """
        if self.total_gt_count == 0: return 0.0

        hits = 0
        for f_idx in range(self.num_frames):
            match_mat = self.match_cache[f_idx]
            if match_mat is None: continue 
            
            # 直接切片前 k 行 (因为输入已排序)
            # match_mat shape: (N_preds, M_gts)
            top_k_mat = match_mat[:k, :] 
            
            # 检查是否有任意一列 (GT) 被击中 (any row is True)
            if top_k_mat.size > 0:
                hits += np.sum(np.any(top_k_mat, axis=0))

        return hits / self.total_gt_count

    #  功能函数 2: 计算 AR @ Threshold (例如 Score > 0.5)
    def calculate_ar_threshold(self, threshold=0.5):
        """
        计算 AR (Score > Threshold)
        """
        if self.total_gt_count == 0: return 0.0

        hits = 0
        for f_idx in range(self.num_frames):
            match_mat = self.match_cache[f_idx]
            if match_mat is None: continue
            
            scores = self.pred_scores_np[f_idx]
            
            # 找到满足阈值的行索引
            valid_mask = scores > threshold
            
            if np.any(valid_mask):
                # 只看满足阈值的那些预测点
                valid_rows = match_mat[valid_mask, :]
                # 统计有多少个 GT 被击中
                hits += np.sum(np.any(valid_rows, axis=0))

        return hits / self.total_gt_count

    #  功能函数 3: 计算 AP (包含 Threshold 过滤)
    def calculate_ap(self, min_score_threshold=0.0):
        """
        计算 AP (Global)。
        如果设置 min_score_threshold > 0，则相当于计算 AP @ Score > Th。
        """
        if self.total_gt_count == 0: return 0.0
        
        # 1. 过滤：只处理全局 Score > 阈值的点
        # 利用 numpy mask 快速筛选
        if min_score_threshold > 0:
            mask = self.all_preds_sorted[:, 0] > min_score_threshold
            if not np.any(mask): return 0.0
            valid_preds = self.all_preds_sorted[mask]
        else:
            valid_preds = self.all_preds_sorted

        num_preds = len(valid_preds)
        tp = np.zeros(num_preds, dtype=np.float32)
        fp = np.zeros(num_preds, dtype=np.float32)
        used_gts = set() # 记录已匹配的 Global Unique ID

        # 2. 匹配逻辑 (贪婪)
        for i in range(num_preds):
            score, f_idx, idx_in_frame = valid_preds[i]
            f_idx, idx_in_frame = int(f_idx), int(idx_in_frame)
            
            match_mat = self.match_cache[f_idx]
            
            hit_gt = False
            if match_mat is not None:
                # 获取该预测点命中的所有 GT 索引
                matched_gt_indices = np.where(match_mat[idx_in_frame])[0]
                
                for gt_idx in matched_gt_indices:
                    uid = self.gt_ids[f_idx][gt_idx]
                    if uid not in used_gts:
                        used_gts.add(uid)
                        tp[i] = 1.0
                        hit_gt = True
                        break # 命中一个未使用的即可
            
            if not hit_gt:
                fp[i] = 1.0
        
        # 3. 计算 AUC
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        precisions = cum_tp / (cum_tp + cum_fp + 1e-16)
        recalls = cum_tp / self.total_gt_count
        
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]
        
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return ap

    #  功能函数 4: 计算 Recall @ Fixed FPPI
    def calculate_recall_at_fppi(self, fppi_limit):
        """
        计算 Recall @ Fixed FPPI (False Positives Per Image).
        
        公式:
        Total Allowed FPs = fppi_limit * num_frames
        
        :param fppi_limit: 平均每张图允许的误报数量 (float)
        """
        if self.total_gt_count == 0: return 0.0
        if self.num_frames == 0: return 0.0
        
        # 1. 计算总误报预算 (Budget)
        # 例如: 1000 帧视频，FPPI=0.1 -> 允许总共 100 个误报
        # 例如: 1000 帧视频，FPPI=100 -> 允许总共 100,000 个误报
        fp_budget = fppi_limit * self.num_frames
        
        current_tp = 0
        current_fp = 0
        used_gts = set()
        
        # 2. 遍历全局排序后的预测 (从最高置信度开始)
        # 必须确保 self.all_preds_sorted 已经存在 (在 __init__ 中生成的)
        for i in range(len(self.all_preds_sorted)):
            score, f_idx, idx_in_frame = self.all_preds_sorted[i]
            f_idx, idx_in_frame = int(f_idx), int(idx_in_frame)
            
            # --- 判定 TP / FP ---
            match_mat = self.match_cache[f_idx]
            is_tp = False
            
            if match_mat is not None:
                # 获取该点命中的 GT 索引
                matched_gt_indices = np.where(match_mat[idx_in_frame])[0]
                
                # 贪婪匹配：只要命中一个未被匹配的 GT 就算 TP
                for gt_idx in matched_gt_indices:
                    uid = self.gt_ids[f_idx][gt_idx]
                    if uid not in used_gts:
                        used_gts.add(uid)
                        is_tp = True
                        break 
            
            # --- 累计计数 ---
            if is_tp:
                current_tp += 1
            else:
                current_fp += 1
            
            # --- 检查是否耗尽预算 ---
            # 一旦当前误报数超过了 (帧数 * FPPI)，立即停止
            if current_fp >= fp_budget:
                break
        
        # 3. 计算 Recall
        return current_tp / self.total_gt_count

    #  功能函数 5: 计算 Recall @ threshold
    def calculate_recall_threshold(self, threshold=0.5):
        """
        计算 Recall @ Score > Threshold。
        """
        if self.total_gt_count == 0: return 0.0

        hits = 0
        for f_idx in range(self.num_frames):
            match_mat = self.match_cache[f_idx]
            if match_mat is None: continue
            
            scores = self.pred_scores_np[f_idx]
            
            # 找到满足阈值的行索引
            valid_mask = scores > threshold
            
            if np.any(valid_mask):
                # 只看满足阈值的那些预测点
                valid_rows = match_mat[valid_mask, :]
                # 统计有多少个 GT 被击中
                hits += np.sum(np.any(valid_rows, axis=0))

        return hits / self.total_gt_count

    #  Wrapper: 一键评估所有指标
    def run_full_evaluation(self):
        print("Running Evaluation...")
        
        ar_at_top100 = self.calculate_ar_top_k(100)
        recall_at_100FPs = self.calculate_recall_at_fppi(100)
        ar_at_0 = self.calculate_ar_threshold(0)
        

        results = {
            'AR@Top100': ar_at_top100,
            'Recall@100FPs': recall_at_100FPs,
            'AR@0.0': ar_at_0,
        }
        
        print("-" * 30)
        for k, v in results.items():
            print(f"{k:<15}: {v:.4f}")
        print("-" * 30)
        
        return results


    def analyze_search_performance(self, score_threshold=0.5):
        """
        分析搜索任务的性能：计算每个目标何时首次被发现。
        
        :param score_threshold: 只有分数 > threshold 的预测才算有效的“发现”
        :return: dict 包含详细统计和平均指标
        """
        if self.total_gt_count == 0:
            return {}

        # 1. 记录每个 Track ID 的生命周期 (GT Info)
        # 结构: { 'raw_track_id': {'start_frame': 9999, 'end_frame': -1} }
        gt_track_info = {}
        
        # 为了映射方便，我们需要从 unique_id 反推 raw_track_id
        # 或者在遍历时直接读取 raw id。这里我们需要重新遍历一下 GT 列表获取 Raw ID
        # 因为 __init__ 里存的是 unique_id
        
        # 我们可以直接遍历 self.gt_ids，假设 unique_id 格式是 "{frame_idx}_{track_id}"
        # 但为了稳健，最好在 __init__ 里存一下 raw_track_id，或者利用现有的结构解析
        
        # 临时遍历一次 GT 建立 Track ID 索引 (耗时极短)
        # 我们需要知道每个目标最早出现在哪一帧
        for f_idx, ids_list in enumerate(self.gt_ids):
            for uid in ids_list:
                # 解析 Raw Track ID (假设 uid 格式为 "frame_id_raw_id")
                # 注意：这里需要根据你之前的 uid 生成逻辑 split
                # 之前代码: uid = f"{f_idx}_{item.get('track_id', i)}"
                raw_track_id = uid.split('_', 1)[1] 
                
                if raw_track_id not in gt_track_info:
                    gt_track_info[raw_track_id] = {'start': f_idx, 'detected_at': None}
                
                # 没必要更新 end，因为我们只关心 start

        # 2. 遍历每一帧，寻找首次匹配
        total_tracks = len(gt_track_info)
        detected_tracks_count = 0
        
        for f_idx in range(self.num_frames):
            match_mat = self.match_cache[f_idx]
            if match_mat is None: continue
            
            scores = self.pred_scores_np[f_idx]
            
            # 筛选分数达标的预测
            valid_mask = scores > score_threshold
            if not np.any(valid_mask): continue
            
            # 只看高分预测的匹配情况
            valid_match_mat = match_mat[valid_mask, :]
            
            # 这一帧被击中的所有 GT 的索引 (去重)
            # any(axis=0) 表示某一列(GT)是否被任意一行(Pred)击中
            hit_gt_indices = np.where(np.any(valid_match_mat, axis=0))[0]
            
            for gt_idx in hit_gt_indices:
                # 获取该 GT 的 Unique ID
                uid = self.gt_ids[f_idx][gt_idx]
                raw_track_id = uid.split('_', 1)[1]
                
                # 检查该目标是否是“首次”被发现
                if gt_track_info[raw_track_id]['detected_at'] is None:
                    gt_track_info[raw_track_id]['detected_at'] = f_idx
                    detected_tracks_count += 1
        
        # 3. 计算统计指标
        delays = []
        detection_frame_indices = []
        
        for tid, info in gt_track_info.items():
            if info['detected_at'] is not None:
                # 搜索延迟 = 发现时刻 - 目标首次出现时刻
                delay = info['detected_at'] - info['start']
                delays.append(delay)
                detection_frame_indices.append(info['detected_at'])
        
        # 平均搜索延迟 (Mean Search Delay)
        mean_delay = np.mean(delays) if delays else 0.0
        
        # 发现率 (Detection Rate / Search Success Rate)
        success_rate = detected_tracks_count / total_tracks if total_tracks > 0 else 0.0

        print("-" * 30)
        print(f"Search Performance (@Conf>{score_threshold})")
        print(f"Total Targets:     {total_tracks}")
        print(f"Found Targets:     {detected_tracks_count}")
        print(f"Success Rate:      {success_rate:.2%}")
        print(f"Mean Search Delay: {mean_delay:.2f} frames")
        print("-" * 30)

        return {
            'success_rate': success_rate,
            'mean_delay': mean_delay,
            'track_details': gt_track_info # 包含每个 ID 的详细数据，方便后续画图
        }


def get_annotation_by_frame_id():
    with open(os.path.join(ITEM_DIR, 'maritime', 'annotations', 'instances_train_objects_in_water.json'), 'r') as f:
        data = json.load(f)
        annotation1 = data['annotations']
    with open(os.path.join(ITEM_DIR, 'maritime', 'annotations', 'instances_val_objects_in_water.json'), 'r') as f:
        data = json.load(f)
        annotation2 = data['annotations']

    raw_annotations = annotation1 + annotation2
    sort_annotations = sorted(raw_annotations, key=lambda x: x['image_id'])
    annotations_by_frame_id = [[] for _ in range(sort_annotations[-1]['image_id'] + 1)]
    for ann in sort_annotations:
        if ann['category_id'] == 1 or ann['category_id'] == 2:
            frame_id = ann['image_id']
            annotations_by_frame_id[frame_id].append(ann)

    return annotations_by_frame_id


def get_top_k_torch(response_tensor, k=1000):
    """
    输入: response_tensor: (H, W) 的 torch.Tensor (可以是 CUDA)
    输出: list of (y, x, score)
    """
    # 1. 获取形状
    H, W = response_tensor.shape[-2:]
    
    # 防止 k 超过像素总数
    k = min(k, H * W)

    # 2. Flatten (展平)
    flat_response = response_tensor.view(-1)

    # 3. TopK 核心操作
    # torch.topk 默认就是降序 (largest=True)，且只找前 k 个，速度极快
    top_n_values, top_n_indices = torch.topk(flat_response, k=k)

    # 4. Unravel Index (计算坐标)
    top_n_y = torch.div(top_n_indices, W, rounding_mode='floor') # y (row)
    top_n_x = top_n_indices % W                                  # x (col)

    # 5. 格式转换 (适配之前的 evaluator)
    if response_tensor.is_cuda:
        top_n_y = top_n_y.cpu()
        top_n_x = top_n_x.cpu()
        top_n_values = top_n_values.cpu()

    # 转换为 Python list of tuples: [(y, x, val), ...]
    targets = list(zip(top_n_y.tolist(), top_n_x.tolist(), top_n_values.tolist()))
    
    return targets


def run_and_evaluate_in_CPU(sequence_iterator, annotations, start_frame, end_frame):

    ''' Model instantiation '''
    objModel = instancing_model('vSTMD_F_L', device='cpu') 
    objModel.init_config()


    totalTime = 0
    frame_idx = start_frame
    responses = [None for _ in range(end_frame - start_frame)]
    temp_response = []

    '''Run inference'''
    for frame_idx in tqdm(range(start_frame, end_frame)):

        # Get the next frame from the input source
        gray_img, _, cap = sequence_iterator.get_next_frame()
        if gray_img is None: break
        
        # Perform inference using the model
        result, runTime = inference(objModel, gray_img)
        totalTime += runTime
        
        temp_response.append({'frame_idx':frame_idx,
                              'res': result['response']})
        
        if len(temp_response) > 100:
            for item in temp_response:
                frame_idx = item['frame_idx']
                result_response = torch.from_numpy(item['res']).float().to('cuda').unsqueeze(0).unsqueeze(0)

                # Normalize and NMS
                max_response = torch.max(result_response) 
                if max_response > 0:
                    result_response /= max_response
                response = nms(result_response, device='cuda')
                targets = get_top_k_torch(response, k=1000)
                responses[frame_idx-start_frame] = targets
            temp_response = []

    # 处理剩余的帧
    for item in temp_response:
        frame_idx = item['frame_idx']
        result_response = torch.from_numpy(item['res']).float().to('cuda').unsqueeze(0).unsqueeze(0)

        # Normalize and NMS
        max_response = torch.max(result_response) 
        if max_response > 0:
            result_response /= max_response
        response = nms(result_response, device='cuda')
        targets = get_top_k_torch(response, k=1000)
        responses[frame_idx-start_frame] = targets
    temp_response = []
        
        
    # Evaluate
    evaluator = FastVideoEvaluator(responses, annotations[start_frame:end_frame])
    res = evaluator.run_full_evaluation()

    search_res = evaluator.analyze_search_performance(score_threshold=0.2)

    FPS = (end_frame - start_frame) / totalTime

    print("-" * 30)
    print(f"Processing FPS: {FPS:.2f}")
    print("-" * 30)

    return res, search_res, FPS


def run_and_evaluate_in_GPU(sequence_iterator, annotations, start_frame, end_frame):

    ''' Model instantiation '''
    objModel = instancing_model('vSTMD_F_L', device='cuda') 
    objModel.init_config()


    totalTime = 0
    frame_idx = start_frame
    responses = [None for _ in range(end_frame - start_frame)]

    '''Run inference'''
    for frame_idx in tqdm(range(start_frame, end_frame)):

        # Get the next frame from the input source
        gray_img, _, cap = sequence_iterator.get_next_frame()
        if gray_img is None: break

        gray_img = torch.from_numpy(gray_img).float().to('cuda').unsqueeze(0).unsqueeze(0)
        
        # Perform inference using the model
        time_start = time.time()
        result, runTime = inference(objModel, gray_img)
        torch.cuda.synchronize()
        totalTime += time.time() - time_start
        
        result_response = result['response']
        # Normalize and NMS
        max_response = torch.max(result_response) 
        if max_response > 0:
            result_response /= max_response
        response = nms(result_response, device='cuda')
        targets = get_top_k_torch(response, k=1000)
        responses[frame_idx-start_frame] = targets

    # Evaluate
    evaluator = FastVideoEvaluator(responses, annotations[start_frame:end_frame])
    res = evaluator.run_full_evaluation()

    search_res = evaluator.analyze_search_performance(score_threshold=0.2)

    FPS = (end_frame - start_frame) / totalTime

    print("-" * 30)
    print(f"Processing FPS: {FPS:.2f}")
    print("-" * 30)

    return res, search_res, FPS
    

def main():
    video_map = {
        'demo1': {'name': 'demo1-SeaDronesSee-696-1410.mp4', 
                  'start_frame': 696, 'end_frame': 1410},
        'demo2': {'name': 'demo2-SeaDronesSee-1697-2411.mp4',
                    'start_frame': 1697, 'end_frame': 2411},
        'demo3': {'name': 'demo3-SeaDronesSee-3666-4166.mp4',
                    'start_frame': 3666, 'end_frame': 4166},
        'demo4': {'name': 'demo4-SeaDronesSee-22931-23545.mp4',
                    'start_frame': 22931, 'end_frame': 23545},
        'demo5': {'name': 'demo5-SeaDronesSee-29713-30312.mp4',
                    'start_frame': 29713, 'end_frame': 30312},
    }

    ar_at_top100_list = []
    recall_at_100FPs_list = []
    ar_at_0_list = []
    success_rate_list = []
    mean_delay_list = []
    track_details_list = []
    FPS_list = []

    
    for key, value in video_map.items():

        sequence_iterator = FrameIterator(os.path.join(ITEM_DIR, 'maritime', 'videos', value['name']), 
                                        is_video=True)

        anno = get_annotation_by_frame_id()

        # res, search_res, FPS = run_and_evaluate_in_CPU(sequence_iterator, anno, 
        #                         start_frame=value['start_frame'],
        #                         end_frame=value['end_frame'])
        res, search_res, FPS = run_and_evaluate_in_GPU(sequence_iterator, anno, 
                                start_frame=value['start_frame'],
                                end_frame=value['end_frame'])
        ar_at_top100_list.append(res['AR@Top100'])
        recall_at_100FPs_list.append(res['Recall@100FPs'])
        ar_at_0_list.append(res['AR@0.0'])
        success_rate_list.append(search_res['success_rate'])
        mean_delay_list.append(search_res['mean_delay'])
        track_details_list.append(search_res['track_details'])
        FPS_list.append(FPS)

    mean_ar_at_top100 = np.mean(ar_at_top100_list)
    mean_recall_at_100FPs = np.mean(recall_at_100FPs_list)
    mean_ar_at_0 = np.mean(ar_at_0_list)
    mean_success_rate = np.mean(success_rate_list)
    mean_search_delay = np.mean(mean_delay_list)
    mean_FPS = np.mean(FPS_list)

    
    print("Overall Evaluation Results:")
    print("=" * 30)
    print(f"Mean AR@Top100       : {mean_ar_at_top100:.4f}")
    print(f"Mean Recall@100FPs   : {mean_recall_at_100FPs:.4f}")
    print(f"Mean AR@0.0          : {mean_ar_at_0:.4f}")
    print(f"Mean Search Success Rate : {mean_success_rate:.2%}")
    print(f"Mean Search Delay        : {mean_search_delay:.2f} frames")
    print(f"Mean Processing FPS      : {mean_FPS:.2f} FPS")
    print("=" * 30)

    with open(os.path.join(ITEM_DIR, 'maritime', 'results', 'demos_evaluation_results.json'), 'w') as f:
        json.dump({'AR@Top100': ar_at_top100_list,
                    'Recall@100FPs': recall_at_100FPs_list,
                    'AR@0.0': ar_at_0_list,
                    'Search Success Rate': success_rate_list,
                    'mean Search Delay': mean_delay_list,
                    'mean_AR@Top100': mean_ar_at_top100,
                    'mean_Recall@100FPs': mean_recall_at_100FPs,
                    'mean_AR@0.0': mean_ar_at_0,
                    'mean_Search Success Rate': mean_success_rate,
                    'mean_Mean Search Delay': mean_search_delay,
                    'mean_FPS': mean_FPS,
                    'Track Details': track_details_list,
                    },
            f, indent=4)



    
    

if __name__ == '__main__':
    main()
