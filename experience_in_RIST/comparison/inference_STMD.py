import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import concurrent.futures
from tqdm import tqdm

import config_task
from config_task import stmdModelList, datasetInfo, ristDatasetPath, modelOptFolder
from model_setting import modelParas
from smalltargetmotiondetectors.api import evaluate # type: ignore


def main_infer_STMD():

    for datasetName in tqdm(datasetInfo.keys()):
        # Dataset path
        inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
        # Output path
        outputFolder = os.path.join(modelOptFolder, datasetName)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
 
        for modelName in tqdm(stmdModelList, desc=f'Processing dataset: {datasetName}', leave=False):

            inferOpt, inferDire, totalRunningTime = evaluate.inference_task(
                    modelName, 
                    inputpath = inputPath, 
                    inputType = 'VidstreamReader', 
                    startFrame = 0, 
                    endFrame = len(datasetInfo[datasetName]), 
                    **modelParas[datasetName]['para'+modelName])

            # Save results
            with open(os.path.join(outputFolder, f'{modelName}_result.json'), 'w') as f:
                saveData = {
                    'response'  : inferOpt,
                    'direction' : inferDire,
                    'runningtime'   : totalRunningTime,
                    }
                json.dump(saveData, f)   



def _task(datasetName, modelName):

    inferOpt, inferDire, totalRunningTime = evaluate.inference_task(
            modelName, 
            inputpath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4'), 
            inputType = 'VidstreamReader', 
            startFrame = 0, 
            endFrame = len(datasetInfo[datasetName]),
            device = 'cuda', 
            **modelParas[datasetName]['para'+modelName])

    # Save results
    with open(os.path.join(modelOptFolder, datasetName, f'{modelName}_result.json'), 'w') as f:
        saveData = {
            'response'  : inferOpt,
            'direction' : inferDire,
            'runningtime'   : totalRunningTime,
            }
        json.dump(saveData, f) 


def quick_infer_STMD():

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for modelName in stmdModelList:
            for datasetName in datasetInfo.keys():      
                futures.append(executor.submit(_task, datasetName, modelName) )   

        for future in tqdm(concurrent.futures.as_completed(futures), 
                            desc='quick inference task',
                            total=len(futures),
                            ):
            future.result()


    

if __name__ == "__main__":
    from datetime import datetime
    
    print("start time:", datetime.now())

    main_infer_STMD()
    # quick_infer_STMD()

    print("end time:", datetime.now())
