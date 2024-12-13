import os
import sys
sys.path.append(os.path.abspath(__file__))
sys.path.append(
    os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
        )))

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import json
import concurrent.futures
from tqdm import tqdm

import add_package_path 
from smalltargetmotiondetectors.util.evaluate_module import evaluation_model_by_video # type: ignore


MODEL_LIST = ['ZBS', 'RPFC']
FPS_LIST = [60, 120, 240]
datasetInfo = {
    'GX010071-1': list(range(1300)),
    'GX010220-1': list(range(1300)),
    'GX010228-1': list(range(1300)),
    'GX010230-1': list(range(2400)),
    'GX010231-1': list(range(2400)),
    'GX010241-1': list(range(3600)),
    'GX010250-1': list(range(2000)),
    'GX010266-1': list(range(2400)),
    'GX010290-1': list(range(1300)),
    'GX010291-1': list(range(1300)),
    'GX010303-1': list(range(2400)),
    'GX010307-1': list(range(1000)),
    'GX010315-1': list(range(1000)),
    'GX010321-1': list(range(1000)),
    'GX010322-1': list(range(1300)),
    'GX010327-1': list(range(900)),
    'GX010335-1': list(range(1300)),
    'GX010336-1': list(range(1000)),
    'GX010337-1': list(range(700)),
}

ristDatasetPath = os.path.join('D:\STMD_Dataset', 'RIST')

modelOptFolder = os.path.join('D:\STMD_Dataset', 'evaluate_RIST')




def _task(modelOpt, groundTruth, gTError, startFrame, endFrame, timePerImage, outPath):
    listTP, listFN, listFP = evaluation_model_by_video(modelOpt[startFrame:endFrame], 
                            groundTruth[startFrame:endFrame], 
                            confidenceThreshold = 0.5, 
                            gTError = gTError)
    
    # only a small moving target per frame
    TP = sum(1 for x in listTP if x != 0)
    FN = sum(1 for x in listFN if x != 0)
    FP = sum(listFP) / 100 # 100 is the average pixel area of a small moving target

    recall = TP / (TP + FN) if (TP + FN) != 0 else 1.0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 1.0
    f1_score = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0.0

    # 创建父文件夹
    os.makedirs(os.path.dirname(outPath), exist_ok=True)

    # 写入JSON文件
    with open(outPath, 'w') as json_file:
        json.dump({'AUC': -1, 'AR': recall, 'AP': precision, 'F1': f1_score, 'timePerImage': timePerImage}, 
                  json_file, indent=2)


def main_evalu_STMD():
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []

        for oName in datasetInfo.keys():
            for fps in FPS_LIST:
                if fps == 240:
                    datasetName = oName
                else:
                    datasetName = f'{oName}_{fps}Hz'

                for modelName in MODEL_LIST:
                    # import inference result
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

                    outPath = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'result', f'RIST_{fps}Hz', datasetName, modelName + 'evaluate.json'
                        )
                    
                    if modelName == 'ZBS':
                        startFrame =  int(len(datasetInfo[oName])*fps/240 * 0.1)
                    else:  
                        startFrame = 0
                    future = executor.submit(_task, 
                                            inferResult, bboxData, 1, 
                                            startFrame, 
                                            int(len(datasetInfo[oName])*fps/240),
                                            timePerImage, outPath)
 

                    futures.append(future)   


        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            desc='evaluate task',
            total=len(datasetInfo) * len(MODEL_LIST) * len(FPS_LIST)
            ):

            future.result()


def debug_mode_signal_process():
    datasetName = 'GX010220-1'
    fps = 240
    modelName = 'RPFC'
                
    # import inference result
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

    outPath = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'result', f'RIST_{fps}Hz', datasetName, modelName + 'evaluate.json'
        )
    
    _task(inferResult, bboxData, 2, 
            0, 
            len(datasetInfo[datasetName]),
            timePerImage, outPath)


 


if __name__ == "__main__":
    from datetime import datetime
    
    print("start time:", datetime.now())

    main_evalu_STMD()
    # debug_mode_signal_process()

    print("end time:", datetime.now())