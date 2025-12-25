import os

import json

# 设置路径
ITEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset_indices():
    """
    加载所有 JSON 数据并构建索引 (只运行一次)
    """
    json_paths = [
        os.path.join(ITEM_DIR, 'maritime', 'annotations', 'instances_train_objects_in_water.json'),
        os.path.join(ITEM_DIR, 'maritime', 'annotations', 'instances_val_objects_in_water.json')
    ]

    frame_meta_map = {}
    image_anno_map = {}

    print("正在加载数据集索引...")
    
    for path in json_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            
            # 1. Images
            if 'images' in data:
                for img in data['images']:
                    f_idx = img.get('id')
                    speed = 0.0
                    if 'meta' in img and 'speed' in img['meta']:
                        speed = img['meta']['speed']
                    if f_idx is not None:
                        frame_meta_map[f_idx] = speed

            # 2. Annotations
            if 'annotations' in data:
                for ann in data['annotations']:
                    # 过滤 category 1 和 2
                    if ann['category_id'] not in [1, 2]:
                        continue
                    
                    img_id = ann['image_id']
                    bbox = ann['bbox'] # [x, y, w, h]
                    
                    if img_id not in image_anno_map:
                        image_anno_map[img_id] = []
                    image_anno_map[img_id].append(bbox)

    print("索引构建完成。")
    return frame_meta_map, image_anno_map


def get_data_lists(video_id, frame_meta_map, image_anno_map):
    """
    提取指定视频的数据
    """
    video_map = {
        'demo1': {'start_frame': 696, 'end_frame': 1410},
        'demo2': {'start_frame': 1697, 'end_frame': 2411},
        'demo3': {'start_frame': 3666, 'end_frame': 4166},
        'demo4': {'start_frame': 22931, 'end_frame': 23545},
        'demo5': {'start_frame': 29713, 'end_frame': 30312},
    }

    key = f'demo{video_id}'
    if key not in video_map:
        return [], [], []

    start_frame = video_map[key]['start_frame']
    end_frame = video_map[key]['end_frame']
    
    bbox_list_sequence = []
    speed_list_sequence = []
    frame_ids = []
    
    for f_idx in range(start_frame, end_frame + 1):
        current_speed = 0.0
        current_bboxes = [] # 默认为空列表
        
        if f_idx in frame_meta_map:
            current_speed = frame_meta_map[f_idx]
            
            if f_idx in image_anno_map:
                current_bboxes = image_anno_map[f_idx]
        
        bbox_list_sequence.append(current_bboxes)
        speed_list_sequence.append(current_speed)
        frame_ids.append(f_idx)

    return bbox_list_sequence, speed_list_sequence, frame_ids


if __name__ == '__main__':
    # 1. 预加载数据
    frame_meta, image_anno = load_dataset_indices()

    # 2. 统计处理
    for i in range(1, 6):
        bboxes, speeds, frames = get_data_lists(i, frame_meta, image_anno)
        
        print(f'Video {i} statistics:')
        
        # --- BBox Area Statistics ---
        # 1. 扁平化：提取整个视频中出现的所有 bbox 的面积
        # bbox 格式是 [x, y, w, h]，面积 = w * h
        all_areas = []
        for frame_bbox_list in bboxes:
            for b in frame_bbox_list:
                area = b[2] * b[3] # width * height
                all_areas.append(area)

        if all_areas:
            print(f'  [Area]  Total Objects Found: {len(all_areas)}')
            print('Min, Max, Avg Area of all objects (category 1 and 2), Min, Max, Avg of velocity')
            print(f'{min(all_areas):d} & {max(all_areas):d} & {int(sum(all_areas)/len(all_areas)):d}', end=' & ')
            print(f'{min(speeds):.2f} & {max(speeds):.2f} & {sum(speeds)/len(speeds):.2f}')


        else:
            print('  [Area]  No objects (cat 1 or 2) found in this video.')
            
        print('-----------------------------------')