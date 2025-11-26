import os
import sys
import json

import math
import numpy as np


ProjectPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ProjectPath)
import config


from smalltargetmotiondetectors.api import get_visualize_handle # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader # type: ignore


def visualize_GT(ristDatasetPath, datasetName):
    inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
    annoPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}_annotation.json')
    ''' Input '''
    hSteam = VidstreamReader(inputPath)
    
    bboxData = []
    direData = []
    centerData = []
    with open(annoPath, 'r') as f:
        _data1 = json.load(f)
    lastPosi = None
    for frame_data in _data1['frames']:
        # Extract the motion_vector and bbox information for the current frame
        bbox = frame_data['objects']['bbox']
        bboxData.append([bbox, ])  # bbox is in [x, y, w, h] 

        centerPos = frame_data['objects']['center_index']
        centerData.append(centerPos)

        motionVec = frame_data['objects']['motion_vector']
        if motionVec is None or len(motionVec) == 0:
            u, v = 0, 0
        else:
            u, v = motionVec[0], motionVec[1]
        dire = np.arctan2(-v, u) 
        if dire < 0:
            dire += 2 * np.pi
        direData.append(dire)



    
    objVisualize = get_visualize_handle()
    # Run inference
    countT = 0
    while hSteam.hasFrame:
        # Get the next frame from the input source
        _, colorImg = hSteam.get_next_frame()
        bbox = bboxData[countT][0]
        bbox.append(1)
        response = [[centerData[countT][0], centerData[countT][1], 1], ]  
        direction = [[centerData[countT][0], centerData[countT][1], direData[countT]], ]  
        res = {'response': response,
               'direction': direction,}
        objVisualize.show_result(colorImg, res)
        countT += 1





if __name__ == '__main__':
    ristDatasetPath = os.path.join('D:/', 'STMD_Dataset', 'RIST')
    datasetName = 'GX010071-1'
    visualize_GT(ristDatasetPath, datasetName)