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


def _task(modelName, datasetName, startFrame, endFrame):
    # loas inference result
    inferResultPath = os.path.join(modelOptFolder, datasetName, f'{modelName}_result.json')
    with open(inferResultPath, 'r') as f:
        _data = json.load(f)
    inferResult = _data['response']
    timePerImage = _data['runningtime'] / len(inferResult)

    # Load annotations
    bboxData = []
    annoPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}_annotation.json')
    with open(annoPath, 'r') as f:
        _data1 = json.load(f)
    for frame_data in _data1['frames']:
        # Extract the motion_vector and bbox information for the current frame
        bbox = frame_data['objects']['bbox']
        bboxData.append([bbox, ])  # bbox is in [x, y, w, h] 

    aucOfROC, AR, AP = evaluate.evaluate_task(inferResult, bboxData, startFrame=startFrame, endFrame=endFrame, plotFigures=False)
    
    f1_score = 2 * AR * AP / (AR + AP) if (AR + AP) != 0 else 0.0

    # 构建文件路径
    file_name = os.path.join(evaluateResultFolder, f'{datasetName}.json')
    # 创建父文件夹
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # 写入JSON文件
    _update_evaluate_result_json(file_name, modelName,
                                 {  'AUC': aucOfROC,
                                    'AR': AR,
                                    'AP': AP,
                                    'F1': f1_score,
                                    'timePerImage': timePerImage} 
                                )


def main_evaluate_location():

    for modelName in tqdm(stmdModelList+LC_model_list, desc='Evaluating location'):

        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = []

            for datasetName in datasetInfo.keys():
                
                future = executor.submit(_task, 
                                        modelName, datasetName, 0, len(datasetInfo[datasetName])
                                        )   
                futures.append(future)   

            for future in concurrent.futures.as_completed(futures):
                future.result()



if __name__ == "__main__":
    from datetime import datetime
    
    print("start time:", datetime.now())

    main_evaluate_location()

    print("end time:", datetime.now())