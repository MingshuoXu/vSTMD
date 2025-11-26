import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math

import json
import concurrent.futures
from tqdm import tqdm
import numpy as np
from math import atan2

import config_task
from config_task import (opticflowModelList, datasetInfo, ristDatasetPath,
                          modelOptFolder, evaluateResultFolder, directionalStmdList)
from smalltargetmotiondetectors import util # type: ignore
from smalltargetmotiondetectors.api import evaluate # type: ignore


def _task_OF(modelName, datasetName, startFrame, endFrame):
    # loas inference result
    inferResultPath = os.path.join(modelOptFolder, datasetName, f'{modelName}_result.json')
    try:
        with open(inferResultPath, 'r') as f:
            _data = json.load(f)
        direResluts = _data['direction']
        timePerImage = _data['runningtime'] / len(direResluts)
    except FileNotFoundError:
        return ...
    

    # Load annotations
    directions = []
    annoPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}_annotation.json')
    with open(annoPath, 'r') as f:
        _data1 = json.load(f)
    for frame_data in _data1['frames']:
        # Extract the motion_vector and bbox information for the current frame
        if len(frame_data['objects']['motion_vector']):
            u, v = frame_data['objects']['motion_vector']
            # the second axis in image is upward but the second axis of x-y is downward

            _dire = atan2(- v, u)
            if _dire < 0:
                _dire += 2 * np.pi

            directions.append(_dire)

    accAE = 0.0
    for i in range(startFrame, endFrame):
        if direResluts[i] is None:
            continue
        inferDirection = np.array(direResluts[i])
        gtDirection = directions[i-1]
        direErrList = abs(inferDirection[:, -1] - gtDirection)
        direErrList[direErrList > np.pi] = 2 * np.pi - direErrList[direErrList > np.pi]  # ensure the error is within [0, pi]
        sortErrList = np.sort(direErrList)
        meanErr = np.mean(sortErrList[:int(len(sortErrList)* 0.5)]) 
        accAE += meanErr  # Absolute Angular Error
    AAE = accAE / len(direResluts)  # Average Angular Error
    

    # 构建文件路径
    file_path = os.path.join(evaluateResultFolder, datasetName, modelName + 'evaluate.json')
    # 创建父文件夹
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 写入JSON文件
    with open(file_path, 'w') as json_file:
        json.dump({'AAE': AAE, 'timePerImage': timePerImage}, json_file, indent=2)
    
    return ...


def _task_STMD(modelName, datasetName, startFrame, endFrame):
    # loas inference result
    inferResultPath = os.path.join(modelOptFolder, datasetName, f'{modelName}_result.json')
    try:
        with open(inferResultPath, 'r') as f:
            _data = json.load(f)
        direResluts = _data['direction']
        respResults = _data['response']
        timePerImage = _data['runningtime'] / len(respResults)
    except FileNotFoundError:
        return ...
    
    # Load annotations
    bboxData = []
    directions = []
    annoPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}_annotation.json')
    with open(annoPath, 'r') as f:
        _data1 = json.load(f)
    for frame_data in _data1['frames']:
        # Extract the motion_vector and bbox information for the current frame
        
        if len(frame_data['objects']['motion_vector']):
            u, v = frame_data['objects']['motion_vector']
            bbox = frame_data['objects']['bbox']

            bboxData.append(bbox)

            _dire = atan2(- v, u)
            if _dire < 0:
                _dire += 2 * np.pi


            directions.append(_dire)

    def calc_direction_error(respRes, direRes, bbox, gtDire):
        """
        Calculate the absolute angular error between the response results and the ground truth direction.
        """
        x, y, w, h = bbox
        
        filtered_pairs = [(a_row[2], b_row[2]) 
                          for a_row, b_row in zip(respRes, direRes) 
                          if (x - 1 <= a_row[0] <= x + w + 1) and (y - 1 <= a_row[1] <= y + h + 1)
                          ]

        if filtered_pairs:  # 如果有满足条件的元素
            _, dire = max(filtered_pairs, key=lambda x: x[0])
            AE = abs(dire - gtDire)
            # Ensure AAE is in the range [0, pi]
            resAE = AE if AE < np.pi else 2 * np.pi - AE
            return resAE
        else:
            return None
            
    
    accAE = []
    j = 0
    for i in range(startFrame, endFrame):
        if len(respResults[i]) == 0:
            diError = None
        else:
            diError = calc_direction_error(
                respResults[i],
                direResluts[j],
                bboxData[i],
                directions[i-1]
            )
            j += 1
        if diError is not None:
            accAE.append(diError)  
    AAE = np.mean(np.array(accAE))  # Average Angular Error
    AAE = AAE if AAE < np.pi else 2 * np.pi - AAE  # Ensure AAE is in the range [0, pi]

    # 构建文件路径
    file_path = os.path.join(evaluateResultFolder, datasetName, modelName + 'evaluate.json')
    # 创建父文件夹
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 写入JSON文件
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            existing_data = json.load(json_file)
        existing_data.update({
            'AAE': AAE,
            'timePerImage': timePerImage,
        })
        with open(file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=2)
    else:
        with open(file_path, 'w') as json_file:
            json.dump({'AAE': AAE, 'timePerImage': timePerImage}, json_file, indent=2)
    
    return ...


def main_evalu_OF():
    for datasetName in tqdm(datasetInfo.keys()):
        for modelName in opticflowModelList:
            _task_OF(modelName, datasetName, 1, len(datasetInfo[datasetName])-2)   


def main_evalu_STMD():
    for datasetName in tqdm(datasetInfo.keys()):
        for modelName in directionalStmdList:
            _task_STMD(modelName, datasetName, 1, len(datasetInfo[datasetName])-2)
    


if __name__ == "__main__":
    from datetime import datetime
    
    print("start time:", datetime.now())

    # main_evalu_OF()
    main_evalu_STMD()

    print("end time:", datetime.now())