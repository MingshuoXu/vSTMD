import math
import os
import sys

import numpy as np
import json
from tqdm import tqdm

# Add the path 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import smalltargetmotiondetectors as stmd # type: ignore
from smalltargetmotiondetectors.api import instancing_model, inference # type: ignore
from smalltargetmotiondetectors.util.iostream import ImgstreamReader # type: ignore


def process_w_task(w):
    # input
    objIptStream = ImgstreamReader(os.path.join('D:/STMD_Dataset','White-Background', 
                                                    'TW-'+ str(w) +'d-TH-0.8d-TV-50d_s-TL-0-SamFre-200', 
                                                    'WhiteBG*.tif'))

    objSTMDNet = instancing_model('STMDNet')
    objSTMDNet.init_config()


    maxSTMDNet = 0
    while objIptStream.hasFrame:
        # Read the next frame from the image stream
        grayImg, _ = objIptStream.get_next_frame()
        
        resultSTMDNet = inference(objSTMDNet, grayImg)

        maxSTMDNet = max(maxSTMDNet, resultSTMDNet['response'].max())

        if objIptStream.currIdx == 120:
            return maxSTMDNet

def process_h_task(h):
    # input
    objIptStream = ImgstreamReader(os.path.join('D:/STMD_Dataset','White-Background', 
                                                    'TW-0.8d-TH-'+ str(h) +'d-TV-50d_s-TL-0-SamFre-200', 
                                                    'WhiteBG*.tif'))

    objSTMDNet = instancing_model('STMDNet')
    objSTMDNet.init_config()


    maxSTMDNet = 0
    while objIptStream.hasFrame:
        # Read the next frame from the image stream
        grayImg, _ = objIptStream.get_next_frame()
        
        resultSTMDNet = inference(objSTMDNet, grayImg)

        maxSTMDNet = max(maxSTMDNet, resultSTMDNet['response'].max())

        if objIptStream.currIdx == 120:
            return maxSTMDNet

def process_c_task(c):
    # input
    objIptStream = ImgstreamReader(os.path.join('D:/STMD_Dataset','White-Background', 
                                                    'TW-0.8d-TH-0.8d-TV-50d_s-TL-'+str(c)+'-SamFre-200', 
                                                    'WhiteBG*.tif'))

    objSTMDNet = instancing_model('STMDNet')
    objSTMDNet.init_config()


    maxSTMDNet = 0
    while objIptStream.hasFrame:
        # Read the next frame from the image stream
        grayImg, _ = objIptStream.get_next_frame()
        
        resultSTMDNet = inference(objSTMDNet, grayImg)

        maxSTMDNet = max(maxSTMDNet, resultSTMDNet['response'].max())

        if objIptStream.currIdx == 120:
            return maxSTMDNet
        
def main():
    sizeList = [round(i * 0.1, 1) for i in range(1, 10)] + [i for i in range(1, 11)]
    LuminanceList = [0] + [round(i * 0.1, 1) for i in range(1, 11)]
    

    hCurve = [None for i in range(len(sizeList))]
    wCurve = [None for i in range(len(sizeList))]
    contrastCurve = [None for i in range(len(LuminanceList))]

    for j, contrast in tqdm(enumerate(LuminanceList), desc = 'get contrast curve', total=len(LuminanceList)):
        contrastCurve[j] = process_c_task(contrast)

    for i, size in tqdm(enumerate(sizeList), desc = 'get size curve', total=len(sizeList)):
        hCurve[i] = process_h_task(size)
        wCurve[i] = process_w_task(size)
            
    # 将列表打包到一个字典中
    data = {'hCurve': hCurve, 'wCurve': wCurve, 'contrastCurve': contrastCurve}
    # 保存到 JSON 文件
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'size_and_contrast_curve.json'),
              'w') as file:
        json.dump(data, file)

    print('\n\nDone...')

def read_json():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'size_and_contrast_curve.json'),
              'r') as file:
        loaded_data = json.load(file)

    hCurve = loaded_data['hCurve']
    wCurve = loaded_data['wCurve']
    contrastCurve = loaded_data['contrastCurve']
    return hCurve, wCurve, contrastCurve

def visualize():
    hCurve, wCurve, contrastCurve = read_json()
    
    import matplotlib.pyplot as plt

    sizeList = [round(i * 0.1, 1) for i in range(1, 10)] + [i for i in range(1, 11)]
    LuminanceList = [0] + [round(i * 0.1, 1) for i in range(1, 11)]

    fig1, ax1 = plt.subplots()
    ax1.plot(sizeList, hCurve/np.max(hCurve), 'b', label='hCurve')
    ax1.plot(sizeList, wCurve/np.max(wCurve), 'g', label='wCurve')
    ax1.set_xscale('log')
    ax1.set_xlabel('Size (degree)', fontsize=14)
    ax1.set_ylabel('Peak Model Response (normalised)', fontsize=14)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 1)
    ax1.legend()

    fig3, ax3 = plt.subplots()
    curve = contrastCurve/np.max(contrastCurve)
    ax3.plot(LuminanceList, curve[::-1], 'r', label='contrastCurve')
    ax3.set_xlabel('Webor Contrast', fontsize=14)
    ax3.set_ylabel('Peak Model Response (normalised)', fontsize=14)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax1.legend()
    

    plt.show()
        
if __name__ == '__main__':
    main()
    visualize()