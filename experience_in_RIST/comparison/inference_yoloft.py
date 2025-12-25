import os
ITEM_PTH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys



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


sys.path.append(ITEM_PTH)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config_task
from config_task import (datasetInfo, ristDatasetPath,
                         modelOptFolder, evaluateResultFolder)


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


def predict_stream(video_dir, save_dir, batch_size=100):  
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 加载模型
    model = YOLOFT(os.path.join(ITEM_PTH, 'comparison_models',
                                'YOLOFT', 'pth', 'yoloft-L.pt'))
    model.model = model.model.cuda()
    model.model.eval()

    image_paths = get_image_paths(video_dir)
    json_results = []
    
    print(f"Total frames: {len(image_paths)}, Batch size: {batch_size}")

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
            results = model(batch_datas)

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
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {save_path}")


def main():
    for datasetName in datasetInfo.keys():
        video_dir = os.path.join(ristDatasetPath, f'{datasetName}-Imgs')
        save_dir = os.path.join(modelOptFolder, datasetName)
        # predict(video_dir, save_dir)
        predict_stream(video_dir, save_dir)


if __name__ == "__main__":
    main()