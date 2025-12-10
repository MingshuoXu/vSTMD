import os
import sys
import time
import copy
import math

import json
from tqdm import tqdm
import cv2
import numpy as np


NOW_PATH = os.path.dirname(os.path.abspath(__file__))
TOP_PATH = os.path.dirname(NOW_PATH)
sys.path.append(TOP_PATH)

import config
from smalltargetmotiondetectors.util.iostream import ImgstreamReader # type: ignore
from utils import custom_serialize  

FPS = 1000
totalFrame = 3000


def create_groundtruth(velocity):
    '''
    mid_x = 540 - 20
    mid_y = 310/2.0

    max_vel = -15  # 振幅
    K_2 = 2          # 正弦运动的频率因子
    V_T = - v         # 水平速度

    totalFrame = 300
    groundTruth = [None for _ in range(totalFrame)]

    for t in range(totalFrame):
        groundTruth[t] = [[mid_y + max_vel*math.sin(2.0*math.pi*t/1000*K_2)-3, 
                        V_T*t/1000+mid_x-3,
                        5,
                        5]]
    '''
    screenSize = [470, 310]
    mid_x = screenSize[0]/2.0
    mid_y = screenSize[1]/2.0

    a = screenSize[0] / 2.5  # Length of the horizontal half axis of the ellipse
    b = screenSize[1] / 2.5 # Length of the vertical half axis of the ellipse

    # define target position as a function of time
    groundTruth = [{} for _ in range(totalFrame)]

    lastGT = None
    def get_target_position(frame):
        fps = 1000
        t = frame/fps
        omega = velocity / math.sqrt(a**2 * math.sin(t)**2 + b**2 * math.cos(t)**2)
        x = mid_x + a * math.cos(omega * t)
        y = mid_y + b * math.sin(omega * t)
        return (x, y)

    for frame in range(totalFrame):
        GT = get_target_position(frame)

        if frame > 0:
            direction = np.arctan2(-(GT[1] - lastGT[1]), (GT[0] - lastGT[0])) 
            if direction < 0:
                direction += 2*np.pi
        else:
            direction = None
            
        lastGT = copy.deepcopy(GT)
        groundTruth[frame] = {'bbox': [GT[0]-4, GT[1]-4, 7, 7],
                              'direction': direction}
    
    return groundTruth


def save_groundtruth(groundTruth, v):
    os.makedirs(os.path.join(NOW_PATH, 'Bgr_dire=Leftward_v=250'), exist_ok=True)

    with open(os.path.join(NOW_PATH,
                            'Bgr_dire=Leftward_v=250',
                            f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS={FPS}.json'),
              'w') as f:
        groundTruth = custom_serialize(groundTruth, indent=2)
        f.write(groundTruth)
    
    
def show_groundtruth(v):
    

    with open(os.path.join(NOW_PATH,
                            'Bgr_dire=Leftward_v=250',
                            f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS={FPS}.json'),
              'r') as file:
        groundTruth = json.load(file)

    # 设置输入路径
    inputpath = os.path.join('D:/', 'STMD_Dataset', 'vSTMD_Panorama_Stimuli', 'Bgr_dire=Leftward_v=250',
        f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS={FPS}',
        'vSTMD_Panorama_Stimuli*.tif')

    # 创建图像流读取器
    objIptStream = ImgstreamReader(inputpath)

    cv2.namedWindow('visulization')

    # 迭代每一帧
    for t in range(totalFrame):
        _, colorImg = objIptStream.get_next_frame()

        colorImg = cv2.cvtColor(colorImg, cv2.COLOR_RGB2BGR)

        
        # 绘制 groundTruth 数据
        bbox = groundTruth[t]['bbox']
        x, y, w, h = bbox
        colorImg = cv2.rectangle(colorImg, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 1)

        direction = groundTruth[t]['direction']
        if direction is not None:
            arraw_len = 20
            x_end = int(x + arraw_len * math.cos(direction))
            y_end = int(y - arraw_len * math.sin(direction))
            cv2.arrowedLine(colorImg, (int(x+3), int(y+3)), (x_end+3, y_end+3), (0, 0, 255), 1, tipLength=0.2)

        cv2.putText(colorImg, f"Frame: {t}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('visulization', colorImg)

        key = cv2.waitKey(1)
        if key == 27:  # 按下 'Esc' 键退出
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    for v in tqdm(
        list(range(100, 1000, 100)) + list(range(1000, 2000, 200)) + list(range(2000, 10001, 500))
        , desc='Creating Groundtruth'):
        gt = create_groundtruth(v)
        save_groundtruth(gt, v)

    # show_groundtruth(1000)
