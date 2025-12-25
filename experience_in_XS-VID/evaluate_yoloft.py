import os
import sys


import numpy as np
import cv2
import json
from tqdm import tqdm
import torch
import re
from PIL import Image


ITEM_PTH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ITEM_PTH)
from utils import custom_serialize
sys.path.append(os.path.join(ITEM_PTH, 'eexperience_in_XS-VID'))
import config_task
from config_task import (XS_VID_PTH, modelOptFolder, 
                         annotation_path, evaluate_output_folder,
                         get_test_config, updata_json)



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

        self.video_preds_sparse = video_preds_sparse
        self.video_gts = video_gts
        
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
        """利用广播机制计算所有预测框与真值框的 IoU 矩阵"""
        cache = []
        for f_idx in range(self.num_frames):
            # 假设格式均为 [x1, y1, x2, y2]
            preds = np.array(self.video_preds_sparse[f_idx])  # (N, 4)
            gts = np.array(self.gt_boxes_np[f_idx])      # (M, 4)
            
            if len(preds) == 0 or len(gts) == 0:
                cache.append(None)
                continue
            
            # 1. 扩展维度以进行广播计算 (N, 1, 4) 和 (1, M, 4)
            b1 = preds[:, np.newaxis, :]
            b2 = gts[np.newaxis, :, :]
            
            # 2. 计算交集区域 (Intersection) 的坐标
            inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
            inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
            inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
            inter_y2 = np.minimum(b1[..., 3], b2[..., 3])
            
            # 3. 计算交集面积，确保负值（不相交）处理为 0
            inter_w = np.maximum(0, inter_x2 - inter_x1)
            inter_h = np.maximum(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h # (N, M)
            
            # 4. 计算各自的面积
            area_preds = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1]) # (N,)
            area_gts = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])           # (M,)
            
            # 5. 计算并集面积 (Union): Area1 + Area2 - Intersection
            # 注意：这里 area_preds 和 area_gts 也需要通过广播匹配到 (N, M)
            union_area = area_preds[:, np.newaxis] + area_gts[np.newaxis, :] - inter_area
            
            # 6. 计算 IoU，添加 epsilon 防止除以 0
            iou = inter_area / (union_area + 1e-7)
            
            cache.append(iou>0.5) # 返回的是 (N, M) 的 float 矩阵
            
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


def format_detection_results(data):
    if not data:
        return []

    # 1. 找到最大的 image_id 以确定输出列表的长度
    max_id = max(item['image_id'] for item in data)
    
    # 2. 初始化结果列表，每一项对应一个 image_id (帧)
    formatted_list = [[] for _ in range(max_id + 1)]
    
    # 3. 填充数据
    for item in data:
        img_id = item['image_id']
        bbox = item['bbox_xyxy']  # 取得 [x1, y1, x2, y2]
        score = item['score']
        
        # 将 [x1, y1, x2, y2, score] 组合并添加到对应的帧中
        # 使用 *bbox 展开列表，然后添加 score
        detection_entry = [*bbox, score]
        formatted_list[img_id].append(detection_entry)
        
    return formatted_list


def evaluate_in_GPU(model_name, video_name, anno_es, annot_es_move, anno_et, annot_et_move):
    # Evaluate
    with open(os.path.join(modelOptFolder, video_name, f'{model_name}_result.json'), 'r') as f:
        data = json.load(f)
        responses = format_detection_results(data['results'])
        totalTime = data['total_time']

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



if __name__ == '__main__':

    video_names, annos_es, annos_es_move, annos_et, annos_et_move = get_test_config(annotation_path)

    for video_name in tqdm(video_names.keys()):
        evaluate_in_GPU('yoloft-L', video_name, 
                        annos_es[video_name], annos_es_move[video_name],
                        annos_et[video_name], annos_et_move[video_name],)
