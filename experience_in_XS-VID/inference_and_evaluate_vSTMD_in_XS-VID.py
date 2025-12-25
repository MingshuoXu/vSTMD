import os
import sys
import time


import json
from tqdm import tqdm
import numpy as np
import cv2
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_task import (XS_VID_PTH, modelOptFolder, annotation_path, evaluate_output_folder, 
                         get_test_config, updata_json)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import nms
from smalltargetmotiondetectors.api import instancing_model # type: ignore


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
            in_y = (points[:, 0:1] >= boxes[:, 1:2].T-1) & (points[:, 0:1] < boxes[:, 3:4].T+1)
            in_x = (points[:, 1:2] >= boxes[:, 0:1].T-1) & (points[:, 1:2] < boxes[:, 2:3].T+1)
            
            cache.append(in_y & in_x)
        return cache


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


def get_top_k_torch(response_tensor, direction_tensor, k=1000):
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
    if len(direction_tensor) > 0:
        top_n_directions = direction_tensor[0, 0, top_n_y, top_n_x].cpu().tolist()
        directions = list(zip(top_n_y.tolist(), top_n_x.tolist(), top_n_directions))
    else:
        directions = []


    return targets, directions
 

def evaluate_in_GPU(model_name, video_name, anno_es, annot_es_move, anno_et, annot_et_move):
    # Evaluate
    with open(os.path.join(modelOptFolder, video_name, f'{model_name}_result.json'), 'r') as f:
        data = json.load(f)
        responses = data['response']
        totalTime = data['runningtime']

    evaluator_es = FastVideoEvaluator(responses, anno_es)
    ar_es = evaluator_es.calculate_recall_threshold(0.0)

    evaluator_es_move = FastVideoEvaluator(responses, annot_es_move)
    AR_es_move = evaluator_es_move.calculate_recall_threshold(0.0)

    evaluator_et = FastVideoEvaluator(responses, anno_et)
    ar_et = evaluator_et.calculate_recall_threshold(0.0)

    evaluator_et_move = FastVideoEvaluator(responses, annot_et_move)
    AR_et_move = evaluator_et_move.calculate_recall_threshold(0.0)

    FPS = len(responses) / totalTime

    updata_json(video_name, model_name, {
        'FPS': FPS,
        'AR_es': ar_es,
        'AR_es_move': AR_es_move,
        'AR_et': ar_et,
        'AR_et_move': AR_et_move,
    })


def inference_in_GPU(model_name, video_name, video_info):

    model = instancing_model(model_name, device='cuda')
    model.init_config()

    totalTime = 0
    responses = [None for _ in range(len(video_info))]
    directions = [None for _ in range(len(video_info))]
    for i, img_info in enumerate(video_info):
        img_path = os.path.join(XS_VID_PTH, video_name, f"{img_info['img_num'][:-3]}jpg")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img).float().to('cuda').unsqueeze(0).unsqueeze(0)/255.0

        time_start = time.time()
        torch.cuda.synchronize()
        result, _ = model.process(img)
        torch.cuda.synchronize()
        totalTime += time.time() - time_start

        result_response = result['response']
        # Normalize and NMS
        max_response = torch.max(result_response) 
        if max_response > 0:
            result_response /= max_response
        response = nms(result_response, device='cuda')
        targets, dires = get_top_k_torch(response, result['direction'], k=1000)
        responses[i] = targets
        directions[i] = dires

    

    # Save results
    save_output_folder = os.path.join(modelOptFolder, video_name, f'{model_name}_result.json')
    os.makedirs(os.path.join(modelOptFolder, video_name), exist_ok=True)
    with open(save_output_folder, 'w') as f:
        saveData = {
            'response'  : responses,
            'direction' : directions,
            'runningtime'   : totalTime,
            }
        json.dump(saveData, f) 


def main():

    video_names, annos_es, annos_es_move, annos_et, annos_et_move = get_test_config(annotation_path)

    # for model_name in tqdm(['vSTMD', 'vSTMD_F']):    
    #     for video_name, video_info in tqdm(video_names.items(), desc=f'Processing videos for model {model_name}', leave=False): 
            # inference_in_GPU(model_name, video_name, video_info) 


    for model_name in tqdm(['vSTMD', 'vSTMD_F', 'FracSTMD']):    
        for video_name, video_info in tqdm(video_names.items(), desc=f'Processing videos for model {model_name}', leave=False): 
            evaluate_in_GPU(model_name, video_name, 
                            annos_es[video_name], annos_es_move[video_name],
                            annos_et[video_name], annos_et_move[video_name],) 


def show_annotation(number=0):

    video_names, _, anns_move = get_test_config(annotation_path)

    video_name = list(video_names.keys())[number]
    video_info_annos = anns_move[video_name]
    video_info_images = video_names[video_name]
    cv2.namedWindow('Annotation', cv2.WINDOW_NORMAL)

    for i, frame_annos in enumerate(video_info_annos):
        img_path = os.path.join(XS_VID_PTH, video_name, f"{video_info_images[i]['img_num'][:-3]}jpg")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        for item in frame_annos:
            x, y, w, h = map(int, item['bbox'])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, str(item['track_id']), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Annotation', img)
        key = cv2.waitKey(33)  # 改为 33ms (约30fps) 自动切换，或按 ESC 退出
        if key == 27:  # 按下 ESC 键退出
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # show_annotation()
    main()
