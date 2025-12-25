import os
import sys
import platform

import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils import custom_serialize


if platform.system() == 'Windows':
    XS_VID_PTH = os.path.join('D:/', 'STMD_Dataset', 'XS-VID', 'images')
    modelOptFolder = os.path.join('D:/', 'STMD_Dataset', 'inference_XS-VID')
    annotation_path = os.path.join(os.path.dirname(__file__), 'test.json')

elif platform.system() == 'Linux':
    XS_VID_PTH = os.path.join('/mnt', 'windows_D', 'STMD_Dataset', 'XS-VID', 'images')
    modelOptFolder = os.path.join('/mnt', 'windows_D', 'STMD_Dataset', 'inference_XS-VID')
    annotation_path = os.path.join(os.path.dirname(__file__), 'test.json')


evaluate_output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluate_result', 'XS-VID')


def get_test_config(annotation_path):
    from collections import defaultdict

    # 1. 加载 JSON 数据
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)

    # 2. 建立图片索引 (Image Map)
    # 目的：通过 image_id 快速找到对应的 video_name 和 img_number，
    # 避免使用 list index (item['image_id'] - 1) 这种不安全的方式。
    image_map = {}
    input_videos = defaultdict(list)

    for item in annotation_data['images']:
        # 假设 file_name 格式为 "video_name/img_number"
        try:
            video_name, img_number = item['file_name'].split('/')
        except ValueError:
            # 防止文件名格式不匹配导致报错
            continue
            
        image_id = item['id']
        
        # 存入 map 方便后续查找
        image_map[image_id] = {
            'video_name': video_name,
            'img_number': img_number 
        }

        # 构建 input_videos 列表
        input_videos[video_name].append({
            'id': image_id, 
            'img_num': img_number
        })

    # 3. 处理 Annotations 并聚合 BBox
    # 结构: annotation[video_name][img_number] = [[x1,y1,w1,h1], [x2,y2,w2,h2], ...]
    annotation_es = defaultdict(lambda: defaultdict(list))
    annotation_es_move = defaultdict(lambda: defaultdict(list))
    annotation_et = defaultdict(lambda: defaultdict(list))
    annotation_et_move = defaultdict(lambda: defaultdict(list))

    for item in annotation_data['annotations']:
        image_id = item['image_id']
        
        # 如果这个 image_id 在我们的图片库里找不到，跳过
        if image_id not in image_map:
            continue

        # 获取对应的视频名和帧号
        info = image_map[image_id]
        video_name = info['video_name']
        img_number = info['img_number']

        # 可选：如果你之前的逻辑是想过滤掉特定的框（例如太小的框），可以在这里加判断
        if item['area'] > 12**2: continue 

        # 将 bbox 加入对应视频、对应帧的列表中
        annotation_es[video_name][img_number].append({'bbox': item['bbox'], 'track_id': item['track_id']})

        if item['track_id'].startswith('move'):
            annotation_es_move[video_name][img_number].append({'bbox': item['bbox'], 'track_id': item['track_id']})

        if item['area'] <= 8**2: 
            annotation_et[video_name][img_number].append({'bbox': item['bbox'], 'track_id': item['track_id']})

            if item['track_id'].startswith('move'):
                annotation_et_move[video_name][img_number].append({'bbox': item['bbox'], 'track_id': item['track_id']})

    final_annotation_es = {}
    final_annotation_es_move = {}
    final_annotation_et = {}
    final_annotation_et_move = {}

    for video_name, frames_list in input_videos.items():
        # 1. 先把该视频下的所有图片按帧号排序
        sorted_images = sorted(frames_list, key=lambda x: int(x['img_num'][:-4]))

        video_anno_es_list = []
        video_anno_es_move_list = []
        video_anno_et_list = []
        video_anno_et_move_list = []

        for img_info in sorted_images:
            frame_key = img_info['img_num']
            
            # 2. 如果没有，给空列表 []
            video_anno_es_list.append(annotation_es[video_name].get(frame_key, []))
            video_anno_es_move_list.append(annotation_es_move[video_name].get(frame_key, []))
            video_anno_et_list.append(annotation_et[video_name].get(frame_key, []))
            video_anno_et_move_list.append(annotation_et_move[video_name].get(frame_key, []))

        final_annotation_es[video_name] = video_anno_es_list
        final_annotation_es_move[video_name] = video_anno_es_move_list
        final_annotation_et[video_name] = video_anno_et_list
        final_annotation_et_move[video_name] = video_anno_et_move_list

    return (dict(input_videos), 
            final_annotation_es, final_annotation_es_move,
            final_annotation_et, final_annotation_et_move)


def updata_json(video_name, model_name, update_data):
    json_path = os.path.join(evaluate_output_folder, f'{video_name}.json')
    os.makedirs(evaluate_output_folder, exist_ok=True)

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_data = json.load(f)

        if model_name in existing_data.keys():
            for key, value in update_data.items():
                existing_data[model_name][key] = value
        else:
            existing_data[model_name] = update_data
    else:
        existing_data = {model_name: update_data}
                
    save_data = custom_serialize(existing_data)
    with open(json_path, 'w') as f:
        f.write(save_data)
