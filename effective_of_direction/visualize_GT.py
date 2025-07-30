import os
import sys
import json

import math
import numpy as np
import copy
from tqdm import tqdm
from prettytable import PrettyTable
import concurrent.futures

ProjectPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectPath)
import stmd_package_path


from smalltargetmotiondetectors.api import instancing_model, inference, get_visualize_handle # type: ignore
from smalltargetmotiondetectors.util.iostream import ImgstreamReader # type: ignore


def create_groundtruth(velocity, timeEnd):
    ''' GT(groundtruth)'''
    screenSize = [470, 310]
    mid_x = screenSize[0]/2.0
    mid_y = screenSize[1]/2.0

    a = screenSize[0] / 2.5  # Length of the horizontal half axis of the ellipse
    b = screenSize[1] / 2.5 # Length of the vertical half axis of the ellipse

    def get_target_position(frame, velocity):
        fps = 1000
        t = frame/fps
        omega = velocity / math.sqrt(a**2 * math.sin(t)**2 + b**2 * math.cos(t)**2)
        x = mid_x + a * math.cos(omega * t)
        y = mid_y + b * math.sin(omega * t)
        return (x, y)

    # groundtruth
    posiGT = np.zeros((timeEnd,2))
    DireGT = np.zeros(timeEnd)
    for tt in range(0, timeEnd):
        GT = get_target_position(tt, velocity)
        posiGT[tt,:] = (GT[0], GT[1])
        if tt > 0:
            _dire = np.arctan2(-(GT[1] - lastGT[1]), (GT[0] - lastGT[0])) 
            if _dire < 0:
                _dire += 2*np.pi
            DireGT[tt] = _dire
        lastGT = copy.deepcopy(GT)

    return posiGT, DireGT


def visualize_GT(velocity, timeEnd):

    ''' Input '''
    hSteam = ImgstreamReader(
        os.path.join('D:\\', 'STMD_Dataset', 'PanoramaStimuli', 'BV-250-Leftward', 
                    'SingleTarget-TW-5-TH-5-TV-'+str(velocity)+'-TL-0-Rightward-Amp-0-Theta-0-TemFre-2-SamFre-1000',
                    'PanoramaStimuli*.tif')
                    )

    posiGT, DireGT = create_groundtruth(velocity, timeEnd)
    
    objVisualize = get_visualize_handle()
    # Run inference
    for countT in range(timeEnd):
        # Get the next frame from the input source
        _, colorImg = hSteam.get_next_frame()
        
        response = [[posiGT[countT, 0], posiGT[countT, 1], 1], ]
        direction = [[posiGT[countT, 0], posiGT[countT, 1], DireGT[countT]], ]  
        res = {'response': response,
               'direction': direction,}
        objVisualize.show_result(colorImg, res)
        if countT % 100 == 0:
            print(f'Processing frame {countT}/{timeEnd}...')





if __name__ == '__main__':
    visualize_GT(2000, 3000)