import os
import sys
import time


from ultralytics.models import YOLOFT
from ultralytics.data.build import build_stream_dataloader,build_movedet_dataset
from torch.utils.data import DataLoader
from ultralytics.cfg import cfg2dict
import numpy as np
import cv2
import json
import imageio
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
                         annotation_path, evaluate_output_folder)




class DictWrapper:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

def get_imgMeta(image_path):
    # Input the image address, get the video name, frame number information and return it
    # example: input: /data/videos/video1/000032.png can get video_name:video1 frame_number:32
    # If your video images are named differently, you can modify the function
    video_name = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    frame_num_string = image_name.split('_')[-1] # Assuming the video name is the first part of the filename separated by '_'
    # Extract numeric parts using regular expressions
    match = re.search(r'\d+', frame_num_string)
    digits = match.group()
    frame_num = int(digits)
    img_metas = {
            "frame_number":frame_num,
            "video_name":video_name,
            "epoch":0, # dont change
            "image_path":image_path,
        }
    return img_metas

def abspath_to_filename(abspath):
    # Input the full image address, 
    # transformed to assess the json in the images field file_name similar format, 
    # used to query the corresponding image_id
    return os.path.join(os.path.basename(os.path.dirname(abspath)), os.path.basename(abspath))


def pad_to_32_multiple(image):
    """
    Pads the input image so that both its width and height are multiples of 32.

    Parameters:
    - image: Original image (NumPy array)

    Returns:
    - padded_image: Image padded to dimensions that are multiples of 32 (NumPy array)
    - padding_info: Dictionary containing the number of pixels padded at the top, bottom, left, and right
    """
    height, width, _ = image.shape

    # Calculate the padding required to make dimensions multiples of 32
    pad_height = (32 - height % 32) % 32  # If height is already a multiple of 32, pad_height is 0
    pad_width = (32 - width % 32) % 32

    # Determine padding amounts for top, bottom, left, and right
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # Pad the image with zeros (black pixels)
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Return the padded image and padding information
    padding_info = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    return padded_image, padding_info

def restore_from_padding(padded_image, padding_info):
    """
    Removes padding from the input image using the provided padding information.

    Parameters:
    - padded_image: The padded image (NumPy array)
    - padding_info: Dictionary containing the number of pixels padded at the top, bottom, left, and right

    Returns:
    - original_image: The original image after removing the padding (NumPy array)
    """
    # 获取填充信息
    top = padding_info['top']
    bottom = padding_info['bottom']
    left = padding_info['left']
    right = padding_info['right']

    # 去除填充，恢复原始图像
    original_image = padded_image[top:padded_image.shape[0]-bottom, left:padded_image.shape[1]-right]

    return original_image

def map_bbox_to_original(padding_info, padded_bbox):
    """
    Maps the detected bbox coordinates from the padded image back to the original image.

    Parameters:
    - padding_info: Dictionary containing the number of pixels padded at the top, bottom, left, and right
    - padded_bbox: Detected bbox on the padded image (x_min, y_min, x_max, y_max)

    Returns:
    - original_bbox: Mapped bbox coordinates on the original image (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = padded_bbox

    # Adjust coordinates using padding information to map back to the original image
    x_min_original = x_min - padding_info['left']
    y_min_original = y_min - padding_info['top']
    x_max_original = x_max - padding_info['left']
    y_max_original = y_max - padding_info['top']

    # Return the mapped bbox coordinates on the original image
    original_bbox = (x_min_original, y_min_original, x_max_original, y_max_original)
    return original_bbox


def get_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def get_first_png_file(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                return os.path.join(root, file)
    ValueError("no .png or .jpg files are found, return None")
    return None 

def image_paths_2_predictdatas(image_paths, eval_json=None):
    datas = []
    if eval_json:
        with open(eval_json, 'r') as f:
            coco_data = json.load(f)
        path_id_maps = {img["file_name"]:img["id"] for img in coco_data["images"]}
        
    for i, path in enumerate(image_paths):
        orige_img = cv2.imread(path)
        padding_img, padding_info = pad_to_32_multiple(orige_img)
        img = np.ascontiguousarray(padding_img.transpose(2, 0, 1)[::-1])
        img = torch.from_numpy(img).unsqueeze(0)
        img_metas = get_imgMeta(path)
        
        if i == 0:
            img_metas["is_first"] = True
        else:
            img_metas["is_first"] = False
            
        if eval_json:
            if abspath_to_filename(path) not in path_id_maps: #images have, ann not
                continue
            else:
                image_id = path_id_maps[abspath_to_filename(path)]
        else:
            image_id = 0
            
        img_metas["padding_info"] = padding_info
        datas.append({
            "im_file": [path], 
            "img": {
                "backbone":img,
                "img_metas":[img_metas],
            },
            "image_id": [image_id],
        })
    
    datas[0]["img"]["img_metas"][0]["is_first"] = True 
    return datas 
    
def predict(video_dir, save_dir):  
    
    input_path = get_first_png_file(video_dir)
    file_name = abspath_to_filename(input_path)
    img_metas = get_imgMeta(input_path)
    print(f"input_path: {input_path}")
    print(f"file_name: {file_name} (check it exists in eval json if you need eval)")
    print(f"frame number: {img_metas['frame_number']}, video name: {img_metas['video_name']}")
        
    json_results = []
    os.makedirs(save_dir, exist_ok=True)
    # Load model
    model = YOLOFT(os.path.join(ITEM_PTH, 'comparison_models',
                                'YOLOFT', 'pth', 'yoloft-L.pt'))  # load a custom model
    model.model = model.model.cuda()
    model.model.eval()
    

    image_paths = get_image_paths(video_dir)
    predict_datas = image_paths_2_predictdatas(image_paths)

    results = model(predict_datas)

    for i,result in enumerate(results):
        for bbox_ in result.boxes:
            bbox_xyxy = [round(float(x), 3) for x in bbox_.xyxy[0]]
            bbox_xyxy = map_bbox_to_original(predict_datas[i]["img"]["img_metas"][0]["padding_info"], bbox_xyxy)
            bbox = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3]-bbox_xyxy[1]]
            json_results.append({
            'image_id': predict_datas[i]["image_id"][0],
            'category_id': int(bbox_.cls[0])+1,
            'bbox': bbox,
            'bbox_xyxy': bbox_xyxy,
            'score': round(float(bbox_.conf[0]), 3)})
        

    
    with open(os.path.join(save_dir, "yoloft-L_result.json"), 'w') as f:
        json.dump(json_results, f, indent=2)


def predict_stream(video_dir, save_dir, batch_size=20):  
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 加载模型
    model = YOLOFT(os.path.join(ITEM_PTH, 'comparison_models',
                                'YOLOFT', 'pth', 'yoloft-L.pt'))
    model.model = model.model.cuda()
    model.model.eval()

    image_paths = get_image_paths(video_dir)
    json_results = []
    
    print(f"Total frames: {len(image_paths)}, Batch size: {batch_size}")

    total_time = 0.0
    # 2. 分批次循环处理
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]
        batch_datas = []
        
        # --- 内部循环：准备当前 Batch 的数据 ---
        for j, path in enumerate(batch_paths):
            global_idx = i + j
            orige_img = cv2.imread(path)
            padding_img, padding_info = pad_to_32_multiple(orige_img)
            
            # 预处理张量
            img_tensor = np.ascontiguousarray(padding_img.transpose(2, 0, 1)[::-1])
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).cuda()
            
            img_metas = get_imgMeta(path)
            # 只有整个视频的第一帧 is_first 为 True，而不是每个 batch 的第一帧
            img_metas["is_first"] = (global_idx == 0)
            img_metas["padding_info"] = padding_info
            
            batch_datas.append({
                "im_file": [path], 
                "img": {
                    "backbone": img_tensor,
                    "img_metas": [img_metas],
                },
                "image_id": global_idx, # 如果有 coco id，在此处映射
            })

        # --- 推理：一次性喂入 batch_size 帧 ---
        with torch.no_grad():
            time_tic = time.time()
            results = model(batch_datas)
            total_time += (time.time() - time_tic)

        # --- 处理当前 Batch 的结果 ---
        for k, result in enumerate(results):
            # 获取该帧对应的 padding 转换信息
            current_padding = batch_datas[k]["img"]["img_metas"][0]["padding_info"]
            
            for bbox_ in result.boxes:
                bbox_xyxy = [round(float(x), 3) for x in bbox_.xyxy[0]]
                # 映射回原图尺寸
                bbox_xyxy = map_bbox_to_original(current_padding, bbox_xyxy)
                bbox = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3]-bbox_xyxy[1]]
                
                json_results.append({
                    'image_id': batch_datas[k]["image_id"],
                    'category_id': int(bbox_.cls[0])+1,
                    'bbox': bbox,
                    'bbox_xyxy': bbox_xyxy,
                    'score': round(float(bbox_.conf[0]), 3)
                })

        # 及时释放当前 batch 的显存占位
        del batch_datas
        torch.cuda.empty_cache() 

    # 3. 保存
    save_path = os.path.join(save_dir, "yoloft-L_result.json")
    with open(save_path, 'w') as f:
        json.dump({'results': json_results, 'total_time': total_time}, f, indent=2)
    print(f"Results saved to {save_path}")




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
    annotation = defaultdict(lambda: defaultdict(list))
    annotation_move = defaultdict(lambda: defaultdict(list))

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
        annotation[video_name][img_number].append({'bbox': item['bbox'], 'track_id': item['track_id']})

        if item['track_id'].startswith('move'):
            annotation_move[video_name][img_number].append({'bbox': item['bbox'], 'track_id': item['track_id']})

    final_annotation = {}
    final_annotation_move = {}

    for video_name, frames_list in input_videos.items():
        # 1. 先把该视频下的所有图片按帧号排序
        sorted_images = sorted(frames_list, key=lambda x: int(x['img_num'][:-4]))

        video_anno_list = []
        video_move_list = []

        for img_info in sorted_images:
            frame_key = img_info['img_num']
            
            # 2. 获取常规标注 (如果没有，给空列表 [])
            video_anno_list.append(annotation[video_name].get(frame_key, []))
            
            # 3. 获取 Move 标注 (如果没有，给空列表 [])
            # 这样就实现了：没有 move 物体的帧，列表里就是空的，而不是 None，且帧数对齐
            video_move_list.append(annotation_move[video_name].get(frame_key, []))

        final_annotation[video_name] = video_anno_list
        final_annotation_move[video_name] = video_move_list

    return dict(input_videos), final_annotation, final_annotation_move



def main():
    video_names, annos, annos_move = get_test_config(annotation_path)

    for video_name, video_info in tqdm(video_names.items(), desc=f'Processing yoloft in XS-VID', leave=False): 
        video_dir = os.path.join(XS_VID_PTH, video_name)
        save_dir = os.path.join(modelOptFolder, video_name)
        # predict(video_dir, save_dir)
        predict_stream(video_dir, save_dir)


if __name__ == "__main__":
    main()