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
    file_name = os.path.join(evaluateResultFolder, f'{datasetName}.json')
    # 创建父文件夹
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # 写入JSON文件
    _update_evaluate_result_json(file_name, modelName,
                                 {'AAE': AAE, 'timePerImage': timePerImage}
                                )


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
    AAE = np.nanmean(np.array(accAE))  # Average Angular Error
    AAE = AAE if AAE < np.pi else 2 * np.pi - AAE  # Ensure AAE is in the range [0, pi]

    # 构建文件路径
    file_name = os.path.join(evaluateResultFolder, f'{datasetName}.json')
    # 创建父文件夹
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # 写入JSON文件
    _update_evaluate_result_json(file_name, modelName,
                                 {'AAE': AAE, 'timePerImage': timePerImage}
                                )


def main(max_workers=12):
    for modelName in tqdm(opticflowModelList, desc='Optic Flow direction evaluation'):
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for datasetName in datasetInfo.keys():
                futures.append(
                    executor.submit(_task_OF, modelName, datasetName, 1, len(datasetInfo[datasetName])-2)   
                )

            for future in concurrent.futures.as_completed(futures):
                future.result()


    for modelName in tqdm(directionalStmdList, desc='STMD direction evaluation'):
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for datasetName in datasetInfo.keys():
                futures.append(
                    executor.submit(_task_STMD, modelName, datasetName, 1, len(datasetInfo[datasetName])-2)   
                )
                
            for future in concurrent.futures.as_completed(futures):
                future.result()

    


if __name__ == "__main__":
    from datetime import datetime
    
    print("start time:", datetime.now())

    main()


    print("end time:", datetime.now())

    