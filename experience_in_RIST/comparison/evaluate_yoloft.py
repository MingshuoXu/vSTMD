import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import concurrent.futures
from tqdm import tqdm

import config_task
from config_task import (stmdModelList, LC_model_list, datasetInfo, ristDatasetPath, 
                         modelOptFolder, evaluateResultFolder)
from smalltargetmotiondetectors import util # type: ignore
from smalltargetmotiondetectors.api import evaluate # type: ignore
from utils import custom_serialize



# update evaluate result json
def _update_evaluate_result_json(file_name, model_name, update_dict):

    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)
    else:
        data = {}


    if model_name in data.keys():
        for key, value in update_dict.items():
            data[model_name][key] = value
    else:
        data[model_name] = update_dict


    data = custom_serialize(data, indent=2)

    with open(file_name, 'w') as f:
        f.write(data)



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
        bbox = item['bbox']  # 取得 [x1, y1, w1, h1]
        score = item['score']
        
        # 使用 *bbox 展开列表，然后添加 score
        detection_entry = [*bbox, score]
        formatted_list[img_id].append(detection_entry)
        
    return formatted_list


def evaluate_yoloft_model(dataset_name, start_frame, end_frame):
    # loas inference result
    inferResultPath = os.path.join(modelOptFolder, dataset_name, f'yoloft-L_result.json')
    with open(inferResultPath, 'r') as f:
        _data = json.load(f)
    inferResult = format_detection_results(_data)

    # Load annotations
    bboxData = []
    annoPath = os.path.join(ristDatasetPath, dataset_name, f'{dataset_name}_annotation.json')
    with open(annoPath, 'r') as f:
        _data1 = json.load(f)
    for frame_data in _data1['frames']:
        # Extract the motion_vector and bbox information for the current frame
        bbox = frame_data['objects']['bbox']
        bboxData.append([bbox, ])  # bbox is in [x, y, w, h] 

    aucOfROC, AR, AP = evaluate.evaluate_task(inferResult, bboxData, startFrame=start_frame, endFrame=end_frame, plotFigures=False)
    
    f1_score = 2 * AR * AP / (AR + AP) if (AR + AP) != 0 else 0.0

    # 构建文件路径
    file_name = os.path.join(evaluateResultFolder, f'{dataset_name}.json')
    # 创建父文件夹
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # 写入JSON文件
    _update_evaluate_result_json(file_name, 'yoloft-L',
                                 {  'AUC': aucOfROC,
                                    'AR': AR,
                                    'AP': AP,
                                    'F1': f1_score,
                                    'timePerImage': -1,} 
                                )


def main_evaluate():

    for datasetName in tqdm(datasetInfo.keys()):
        evaluate_yoloft_model(datasetName, 0, len(datasetInfo[datasetName]))   



if __name__ == "__main__":

    main_evaluate()