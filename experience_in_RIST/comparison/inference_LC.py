import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import time
import numpy as np
import torch
from tqdm import tqdm

import config_task
from config_task import LC_model_list, datasetInfo, ristDatasetPath, modelOptFolder
from model_setting import modelParas
from smalltargetmotiondetectors.api import evaluate # type: ignore
from smalltargetmotiondetectors.util.iostream import (ImgstreamReader, VidstreamReader) # type: ignore
from smalltargetmotiondetectors.util.evaluate_module import (get_ROC_curve_data, compute_AUC, # type: ignore
                                    get_thres_recall_data, compute_AR,
                                    get_P_R_curve_data, compute_AP, )
from smalltargetmotiondetectors.util.matrixnms import MatrixNMS # type: ignore
from smalltargetmotiondetectors.util.compute_module import matrix_to_sparse_list # type: ignore
from comparison_models.LC_neuron import LC11, LC18

def _task(model_name, 
        inputpath, 
        inputType = 'ImgstreamReader', 
        startFrame = 0, 
        endFrame = None, 
        ):
    ''' Dynamically create a video stream reader or other input type '''
    inputModule = globals().get(inputType)
    if inputModule is None:
        raise ValueError(f"Unknown inputType: {inputType}")

    objIptStream = inputModule(inputpath, startFrame, endFrame)

    objNMS = MatrixNMS(15)

    model = globals().get(model_name)(fs=240, device='cuda')

    totalRunningTime = 0
    results = []
    ''' Run '''
    while objIptStream.hasFrame:
        # Read the next frame from the video stream
        grayImg, _ = objIptStream.get_next_frame()
        
        # Perform inference using the model
        time_tic = time.time()
        result = model.process(grayImg)
        totalRunningTime += time.time() - time_tic

        # postprocessing

        # response
        if np.max(result) == 0:
            results.append([])
            continue
        response = objNMS.nms(result)
        maxOpt = np.max(response)
        if maxOpt > 0:
            response /= np.max(response)
            responseListType = matrix_to_sparse_list(response)
        else:
            responseListType = []
        results.append(responseListType)          

    return results, totalRunningTime


def main_infer_LC():

    for datasetName in tqdm(datasetInfo.keys()):
        print(f'\n{datasetName}\t')
        # Dataset path
        inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
        # Output path
        outputFolder = os.path.join(modelOptFolder, datasetName)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
 
        for modelName in LC_model_list:
            print(f'inference {modelName}', end='')

            inferOpt, totalRunningTime = _task(
                    modelName, 
                    inputpath = inputPath, 
                    inputType = 'VidstreamReader', 
                    startFrame = 0, 
                    endFrame = len(datasetInfo[datasetName]))

            print(f'running time: {totalRunningTime:.1f} s\t')

            # Save results
            with open(os.path.join(outputFolder, f'{modelName}_result.json'), 'w') as f:
                saveData = {
                    'response'  : inferOpt,
                    'runningtime'   : totalRunningTime,
                    }
                json.dump(saveData, f)     
    

if __name__ == "__main__":
    from datetime import datetime
    
    print("start time:", datetime.now())

    main_infer_LC()

    print("end time:", datetime.now())