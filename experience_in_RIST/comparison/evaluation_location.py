import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import concurrent.futures
from tqdm import tqdm

import config_task
from config_task import stmdModelList, datasetInfo, ristDatasetPath, modelOptFolder, evaluateResultFolder
from smalltargetmotiondetectors import util # type: ignore
from smalltargetmotiondetectors.api import evaluate # type: ignore


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
    file_path = os.path.join(evaluateResultFolder, datasetName, modelName + 'evaluate.json')
    # 创建父文件夹
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 写入JSON文件
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            existing_data = json.load(json_file)
        # Update the existing data
        existing_data.update({
            'AUC': aucOfROC,
            'AR': AR,
            'AP': AP,
            'F1': f1_score,
            'timePerImage': timePerImage
        })
        with open(file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=2)
    else:
        with open(file_path, 'w') as json_file:
            json.dump({'AUC': aucOfROC, 'AR': AR, 'AP': AP, 'F1': f1_score, 'timePerImage': timePerImage}, json_file, indent=2)


def main_evalu_STMD():
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []

        for datasetName in datasetInfo.keys():
            for modelName in stmdModelList:
                future = executor.submit(_task, 
                                         modelName, datasetName, 0, len(datasetInfo[datasetName])
                                         )   
                futures.append(future)   

        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            desc='evaluate task',
            total=len(datasetInfo) * len(stmdModelList) 
            ):
            future.result()



if __name__ == "__main__":
    from datetime import datetime
    
    print("start time:", datetime.now())

    main_evalu_STMD()

    print("end time:", datetime.now())