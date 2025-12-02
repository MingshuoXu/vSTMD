import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json

import config_task
from config_task import stmdModelList, datasetInfo, ristDatasetPath, modelOptFolder
from model_setting import modelParas
from smalltargetmotiondetectors.api import evaluate # type: ignore

def main_infer_STMD():

    for datasetName in datasetInfo.keys():
        print(f'\n{datasetName}\t')
        # Dataset path
        inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
        # Output path
        outputFolder = os.path.join(modelOptFolder, datasetName)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
 
        for modelName in stmdModelList:
            print(f'{modelName} ', end='')

            inferOpt, inferDire, totalRunningTime = evaluate.inference_task(
                    modelName, 
                    inputpath = inputPath, 
                    inputType = 'VidstreamReader', 
                    startFrame = 0, 
                    endFrame = len(datasetInfo[datasetName]), 
                    **modelParas[datasetName]['para'+modelName])

            print(f'running time: {totalRunningTime:.1f} s\t')

            # Save results
            with open(os.path.join(outputFolder, f'{modelName}_result.json'), 'w') as f:
                saveData = {
                    'response'  : inferOpt,
                    'direction' : inferDire,
                    'runningtime'   : totalRunningTime,
                    }
                json.dump(saveData, f)     
    

if __name__ == "__main__":
    from datetime import datetime
    
    print("start time:", datetime.now())

    main_infer_STMD()

    print("end time:", datetime.now())
