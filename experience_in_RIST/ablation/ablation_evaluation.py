import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math import atan2

import json
import concurrent.futures
from tqdm import tqdm
import numpy as np

import config_task
from config_task import ablationModel, datasetInfo, ristDatasetPath, modelOptFolder, evaluateResultFolder
from smalltargetmotiondetectors import util # type: ignore
from smalltargetmotiondetectors.api import evaluate # type: ignore
from utils import custom_serialize

evaluateModelList = ablationModel + ('vSTMD', 'vSTMD_F')

def update_evaluate_result(dataset_name, model_name, new_data):

    # 构建文件路径
    file_path = os.path.join(evaluateResultFolder, f'{dataset_name}.json')
    # 创建父文件夹
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


    # 写入JSON文件
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            existing_data = json.load(json_file)
        if model_name not in existing_data:
            existing_data[model_name] = new_data
        else:
            for key, value in new_data.items():
                existing_data[model_name][key] = value
    else:
        existing_data = {model_name: new_data}

    save_data = custom_serialize(existing_data, indent=2)
    
    with open(file_path, 'w') as f:
        f.write(save_data)


def _task(modelName, datasetName, startFrame, endFrame):
    # loas inference result
    inferResultPath = os.path.join(modelOptFolder, datasetName, f'{modelName}_result.json')
    with open(inferResultPath, 'r') as f:
        _data = json.load(f)
    inferResult = _data['response']
    direResluts = _data['direction']
    timePerImage = _data['runningtime'] / len(inferResult)

    # Load annotations
    bboxData = []
    directions = []
    annoPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}_annotation.json')
    with open(annoPath, 'r') as f:
        _data1 = json.load(f)
    for frame_data in _data1['frames']:
        # Extract the motion_vector and bbox information for the current frame
        bbox = frame_data['objects']['bbox']
        bboxData.append([bbox, ])  # bbox is in [x, y, w, h] 
        if len(frame_data['objects']['motion_vector']):
            u, v = frame_data['objects']['motion_vector']
            bbox = frame_data['objects']['bbox']

            _dire = atan2(- v, u)
            if _dire < 0:
                _dire += 2 * np.pi


            directions.append(_dire)

    aucOfROC, AR, AP = evaluate.evaluate_task(inferResult, bboxData, startFrame=startFrame, endFrame=endFrame, plotFigures=False)
    
    f1_score = 2 * AR * AP / (AR + AP) if (AR + AP) != 0 else 0.0

    AAE = _directional_task(inferResult, direResluts, bboxData, directions, endFrame)


    # 写入JSON文件
    update_evaluate_result(datasetName, modelName, {'AUC': aucOfROC,
                                                    'AR': AR,
                                                    'AP': AP,
                                                    'F1': f1_score,
                                                    'AAE': AAE,
                                                    'timePerImage': timePerImage}
                                                    )
    

def _directional_task(respResults, direResluts, bboxData, directions, endFrame):


    def calc_direction_error(respRes, direRes, bbox, gtDire):
        """
        Calculate the absolute angular error between the response results and the ground truth direction.
        """
        x, y, w, h = bbox[0]
        
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
    for i in range(endFrame):
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

    return AAE


def main_evalu_STMD():
    for modelName in tqdm(evaluateModelList):
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = []

            for datasetName in datasetInfo.keys():
                futures.append(executor.submit(_task, 
                                                modelName, 
                                                datasetName, 
                                                0, 
                                                len(datasetInfo[datasetName])
                                                ))   
                
        for future in concurrent.futures.as_completed(futures):
            future.done()



if __name__ == "__main__":
    from datetime import datetime
    
    print("start time:", datetime.now())

    main_evalu_STMD()
    # _task('vSTMD_F_without_CDGC', 'GX010071-1', 0, 1300)

    print("end time:", datetime.now())