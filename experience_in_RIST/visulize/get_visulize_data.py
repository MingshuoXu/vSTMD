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

# 构建文件路径
currPth = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(currPth, 'visulize_data.json')
os.makedirs(os.path.dirname(file_path), exist_ok=True)


def update_data(updated_dict):
    """
    Update the JSON file with the provided dictionary.
    """
    with open(file_path, 'r') as json_file:
        existing_data = json.load(json_file)

    existing_data.update(updated_dict)

    with open(file_path, 'w') as json_file:
        json.dump(existing_data, json_file )

def _task_GT(datasetName):
    # Load annotations
    directions = []
    centers = []

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
        centers.append(frame_data['objects']['center_index'])

    update_data({f'{datasetName}_groundtruth': {'location': centers,
                                 'direction': directions,}
    })



    return ...


def _task_OF(modelName, datasetName, startFrame, endFrame):
    # loas inference result
    inferResultPath = os.path.join(modelOptFolder, datasetName, f'{modelName}_result.json')
    try:
        with open(inferResultPath, 'r') as f:
            _data = json.load(f)
        direResluts = _data['direction']
    except FileNotFoundError:
        return ...

    annoPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}_annotation.json')
    centers = []
    direGTs = []
    with open(annoPath, 'r') as f:
        _data1 = json.load(f)
    for frame_data in _data1['frames']:
        _center = frame_data['objects']['center_index']
        _center.append(1)
        centers.append(_center)
        if len(frame_data['objects']['motion_vector']):
            u, v = frame_data['objects']['motion_vector']
            # the second axis in image is upward but the second axis of x-y is downward

            _dire = atan2(- v, u)
            if _dire < 0:
                _dire += 2 * np.pi

            direGTs.append(_dire)

    dires = []
    for i in range(startFrame, endFrame):
        if direResluts[i] is None:
            dires.append(math.nan)
            continue
        inferDirection = np.array(direResluts[i])
        dire = np.mean(inferDirection[:, -1])
        if abs(dire - direGTs[i]) < math.pi:
            dires.append(dire)
        else:
            dires.append(math.nan)


    update_data({f'{datasetName}_{modelName}': {'response': centers,
                                                'directions': dires,}
    })
    
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
    direGTs = []
    annoPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}_annotation.json')
    with open(annoPath, 'r') as f:
        _data1 = json.load(f)
    for frame_data in _data1['frames']:
        # Extract the motion_vector and bbox information for the current frame
        bbox = frame_data['objects']['bbox']
        bboxData.append(bbox)
        if len(frame_data['objects']['motion_vector']):
            u, v = frame_data['objects']['motion_vector']
            # the second axis in image is upward but the second axis of x-y is downward

            _dire = atan2(- v, u)
            if _dire < 0:
                _dire += 2 * np.pi

            direGTs.append(_dire)


    responses = []
    directions = []
    j = 0
    for i in range(startFrame, endFrame):
        if len(respResults[i]) == 0:
            responses.append((-1, -1, -1))
            directions.append(math.nan)  
            continue

        x, y, w, h = bboxData[i]
        
        filtered_pairs = [(a_row[2], b_row[2], a_row[0], a_row[1]) 
                          for a_row, b_row in zip(respResults[i], direResluts[j]) 
                          if (x - 1 <= a_row[0] <= x + w + 1) and (y - 1 <= a_row[1] <= y + h + 1)
                          ]

        if filtered_pairs:  # 如果有满足条件的元素
            _, dire, x, y = max(filtered_pairs, key=lambda x: x[0])

            j += 1
            if abs(dire - direGTs[i]) < np.pi:
                directions.append(dire)  
                responses.append((x, y, 1))  # 1 is a placeholder for confidence
            else:
                responses.append((-1, -1, -1))
                directions.append(math.nan)
        else:
            responses.append((-1, -1, -1))
            directions.append(math.nan)

    update_data({f'{datasetName}_{modelName}': {'response': responses,
                                                'directions': directions}
    })
    
    return ...


def main():
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f)

    for datasetName in tqdm(datasetInfo.keys()):

        _task_GT(datasetName)

        for modelName in opticflowModelList:
            _task_OF(modelName, datasetName, 1, len(datasetInfo[datasetName])-2)   

        for modelName in directionalStmdList:
            _task_STMD(modelName, datasetName, 1, len(datasetInfo[datasetName])-2)
    


if __name__ == "__main__":
    
    main()