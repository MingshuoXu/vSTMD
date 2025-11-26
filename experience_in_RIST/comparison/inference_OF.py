import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import time
from tqdm import tqdm 
import numpy as np

import config_task
from config_task import opticflowModelList, datasetInfo, ristDatasetPath, modelOptFolder
from custom_API import (CustomFlowDiffuser, CustomRAFT, CustomSEA_RAFT,  # type: ignore
                        CustomMemFlow, CustomStreamFlow, CustomDpFlow, 
                        prepared_for_rist, flow_to_ang) 
from smalltargetmotiondetectors.util.iostream import VidstreamReader # type: ignore



def _load_groundtruth(pth):
    # Load annotations
    bboxData = []

    with open(pth, 'r') as f:
        _data1 = json.load(f)
    for frame_data in _data1['frames']:
        # Extract the motion_vector and bbox information for the current frame
        bbox = frame_data['objects']['bbox']
        bboxData.append([bbox, ])  # bbox is in [x, y, w, h] 
    return bboxData

def extract_region_to_angle_list(arr, bbox):
    """提取矩形区域并转换为 [i,j,value] 的列表"""
    angMtx = flow_to_ang(arr)  # Convert flow to angle matrix if needed
    x, y, w, h = bbox  # bbox is in [x, y, w, h] format
    x = max(0, x)  # Ensure x is not negative
    y = max(0, y)  # Ensure y is not negative
    region = angMtx[y:y+h, x:x+w]  
    j_indices = np.arange(x, x+w)  
    i_indices = np.arange(y, y+h)  
    
    # generate i, j grid
    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    # 拼接成 [i,j,value] 的 n*3 数组
    ijk_array = np.column_stack([
        i_grid.ravel(),  
        j_grid.ravel(),  
        region.ravel()   
    ])
    
    return ijk_array.tolist()  # 转换为列表

def _task(modelName, inputpath, startFrame, endFrame, groundtruthPth):
    ''' Instantiate the model '''
    objModel = eval(f'Custom{modelName}()')  # Dynamically create the model instance

    objIptStream = VidstreamReader(inputpath, startFrame, endFrame)

    bboxData = _load_groundtruth(groundtruthPth) 


    ''' Initialize the model '''
    totalRunningTime = 0
    results = []
    directions = []
    ''' Run '''
    count = 0
    while objIptStream.hasFrame:
        # Read the next frame from the video stream
        _, colorImg = objIptStream.get_next_frame()

        frame = prepared_for_rist(colorImg)
        
        # Perform inference using the model
        tic0 = time.time()
        flow = objModel.process(newFrame=frame)
        totalRunningTime += time.time() - tic0

        # postprocessing
        if flow is not None:
            # Assuming bboxData[count][0] is [x, y, w, h]
            
            if isinstance(flow, list):
                for j, f in enumerate(flow):
                    k = len(flow) - 1 - j
                    x, y, w, h = bboxData[count-k][0]
                    bbox = (int(x), int(y), int(w), int(h))
                    if j < len(flow)-1:
                        # j=1 -> idx=-1; j=0 -> idx=-2; 
                        directions[-k] = extract_region_to_angle_list(f, bbox)
                    else:
                        # j=2 -> append
                        directions.append(extract_region_to_angle_list(f, bbox))
            else:
                x, y, w, h = bboxData[count][0]
                bbox = (int(x), int(y), int(w), int(h))
                directions.append(extract_region_to_angle_list(flow, bbox))
        else:
            directions.append(None)
        count += 1
    return directions, totalRunningTime

def main_infer_OP():

    for datasetName in tqdm(datasetInfo.keys(), desc='Processing datasets', total=len(datasetInfo)):

        # Dataset path
        inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
        # Output path
        outputFolder = os.path.join(modelOptFolder, datasetName)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        # Ground truth path
        groundtruthPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}_annotation.json')
 
        for modelName in tqdm(opticflowModelList, leave=False, desc='Processing models', total=len(opticflowModelList)):

            inferDire, totalRunningTime = _task(
                    modelName, 
                    inputpath = inputPath, 
                    startFrame = 0, 
                    endFrame = len(datasetInfo[datasetName]), 
                    groundtruthPth = groundtruthPath
                    )

            print(f'\nTotal running time of {modelName} in {datasetName} is {totalRunningTime:.1f} s\n')

            # Save results
            with open(os.path.join(outputFolder, f'{modelName}_result.json'), 'w') as f:
                saveData = {
                    'response'  : None,
                    'direction' : inferDire,
                    'runningtime'   : totalRunningTime,
                    }
                json.dump(saveData, f)     
    

if __name__ == "__main__":
    from datetime import datetime
    
    print("start time:", datetime.now())

    main_infer_OP()

    print("end time:", datetime.now())