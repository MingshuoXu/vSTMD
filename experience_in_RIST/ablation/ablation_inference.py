import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import concurrent.futures

import json
from tqdm import tqdm

import config_task
from config_task import ablationModel, datasetInfo, ristDatasetPath, modelOptFolder
from smalltargetmotiondetectors.api import evaluate # type: ignore

def main_infer_STMD():

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []

        for datasetName in datasetInfo.keys():
            for modelName in ablationModel:
                future = executor.submit(_task, modelName, datasetName
                                         )   
                futures.append(future)   

        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            desc='evaluate task',
            total=len(futures)
            ):
            future.done()


def _task(modelName, datasetName):     
    # Dataset path
    inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
    # Output path
    outputFolder = os.path.join(modelOptFolder, datasetName)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)


    inferOpt, inferDire, totalRunningTime = evaluate.inference_task(
            modelName, 
            inputpath = inputPath, 
            inputType = 'VidstreamReader', 
            startFrame = 0, 
            endFrame = len(datasetInfo[datasetName]))


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