import os
import sys

import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

NOW_PATH = os.path.dirname(os.path.abspath(__file__))
TOP_PATH = os.path.dirname(NOW_PATH)
sys.path.append(TOP_PATH)

import add_package_path
from smalltargetmotiondetectors.util.iostream import ImgstreamReader # type: ignore

FPS = 200



def create_groundtruth(v):
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
    mid_x = 470/2.0
    mid_y = 310/2.0

    a = 470 / 2.5  # Length of the horizontal half axis of the ellipse
    b = 310 / 2.5 # Length of the vertical half axis of the ellipse
    V_T = v  # velocity



    totalFrame = 500
    # define target position as a function of time
    groundTruth = [None for _ in range(totalFrame)]

    for frame in range(totalFrame):
        t = frame / FPS
        omega = V_T / math.sqrt(a**2 * math.sin(t)**2 + b**2 * math.cos(t)**2)
        x = mid_x + a * math.cos(omega * t)
        y = mid_y + b * math.sin(omega * t)
        groundTruth[frame] = [[x-4,y-5,7,7]]
    
    return groundTruth


def save_groundtruth(groundTruth, v):
    # 将列表打包到一个字典中
    data = {'groundTruth': groundTruth}

    with open(os.path.join(NOW_PATH,
                           'SingleTarget-TW-5-TH-5-TV-'+str(v)+'-TL-0-Rightward-Amp-0-Theta-0-TemFre-2-SamFre-'+str(FPS)+'.json'), 
              'w') as file:
        json.dump(data, file)
    
    
def show_groundtruth(v):

    with open(os.path.join(NOW_PATH,
                           'SingleTarget-TW-5-TH-5-TV-'+str(v)+'-TL-0-Rightward-Amp-0-Theta-0-TemFre-2-SamFre-'+str(FPS)+'.json'), 
              'r') as file:
        data = json.load(file)
        groundTruth = data['groundTruth']

    # 设置输入路径
    inputpath = os.path.join('D:/STMD_Dataset', 'PanoramaStimuli', 'BV-250-Leftward',
        'SingleTarget-TW-5-TH-5-TV-'+str(v)+'-TL-0-Rightward-Amp-0-Theta-0-TemFre-2-SamFre-'+str(FPS),
        'PanoramaStimuli*.tif')

    # 创建图像流读取器
    objIptStream = ImgstreamReader(inputpath)

    # 创建一个图形和坐标轴
    fig, ax = plt.subplots()

    # 迭代每一帧
    for t in range(500):
        _, colorImg = objIptStream.get_next_frame()

        # 清除前一帧的绘制
        ax.clear()

        # 显示图像
        ax.imshow(colorImg, cmap='gray', interpolation='none')
        ax.set_title(f'Frame {t}')

        # 绘制 groundTruth 数据
        if groundTruth[t] is not None:
            for bbox in groundTruth[t]:
                x, y, w, h = bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=0.5, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        # 暂停以显示图像，并刷新绘图窗口
        plt.pause(0.001)

    # 关闭图形窗口
    plt.close()


if __name__ == '__main__':
    for v in tqdm(range(100, 2024, 100), desc='Creating Groundtruth'):
        gt = create_groundtruth(v)
        save_groundtruth(gt, v)

    show_groundtruth(1000)
