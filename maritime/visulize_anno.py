import os
ITEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ITEM_DIR)
import time
import json

import torch
import cv2
import numpy as np

# DEVICE = 'cpu' # 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Add the path to the package containing the models
STMD_PYTHON_PATH = os.path.join('D:/', '11_Code', 'Small-Target-Motion-Detectors', 'python')
sys.path.append(STMD_PYTHON_PATH)
from smalltargetmotiondetectors.api import (instancing_model, inference) # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader, ImgstreamReader # type: ignore

from utils import FrameIterator, nms



def show_annotation(sequence_iterator, annotations, start_frame=0):

    ''' Get visualization handle '''
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    if sequence_iterator.img_width > 1920:
        scale_factor = 1920 / sequence_iterator.img_width
        new_width = 1920
        new_height = int(sequence_iterator.img_height * scale_factor)
        cv2.resizeWindow('Result', new_width, new_height)
    else:
        cv2.resizeWindow('Result', sequence_iterator.img_width, sequence_iterator.img_height)

    save_dir = os.path.join(ITEM_DIR, 'maritime', 'results')
    os.makedirs(save_dir, exist_ok=True) # 确保目录存在
    save_path = os.path.join(save_dir, f'{start_frame}_anno.mp4')
    
    # 注意: sequence_iterator.img_width 需确保存在，否则用 color_img.shape 获取
    video_writer = cv2.VideoWriter(
        save_path, 
        cv2.VideoWriter_fourcc(*'mp4v'),
        30, 
        (sequence_iterator.img_width, sequence_iterator.img_height)
    )
    print(f"Video will be saved to: {save_path}")

    frame_idx = start_frame
    '''Run inference'''
    annos = []
    while True:

        # Get the next frame from the input source
        gray_img, color_img, cap = sequence_iterator.get_next_frame()
        if gray_img is None:
            break
   
        # Draw circles at detected target positions

        for target in annotations[frame_idx]:
            x, y, w, h = target['bbox']
            cv2.rectangle(color_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        cv2.putText(color_img, f'Frame: {frame_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # cv2.imshow('Result', color_img)

        video_writer.write(color_img) # 修正: 写入视频

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC pressed -> attempt graceful exit
            break

        frame_idx += 1



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
        frame_id = ann['image_id']
        annotations_by_frame_id[frame_id].append(ann)

    return annotations_by_frame_id


def main(video_id = 1):
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


    sequence_iterator = FrameIterator(os.path.join(ITEM_DIR, 'maritime', 'videos', video_map[f'demo{video_id}']['name']), 
                                      is_video=True)

    anno = get_annotation_by_frame_id()

    show_annotation(sequence_iterator, anno, start_frame=video_map[f'demo{video_id}']['start_frame'])

    



if __name__ == '__main__':
    main(5)

