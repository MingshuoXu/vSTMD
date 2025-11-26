import math
import os
import sys

import concurrent.futures
import numpy as np
import json
from tqdm import tqdm

# Add the path 
sys.path.append(os.path.join(
    'C:/', 'Users', 'mings', 'OneDrive', '1_Code', 
    '0_GitHub', 'Small-Target-Motion-Detectors', 'python'
    ))
import smalltargetmotiondetectors as stmd # type: ignore
from smalltargetmotiondetectors.api import instancing_model, inference # type: ignore
from smalltargetmotiondetectors.util.iostream import ImgstreamReader # type: ignore


def process_Frac_task(v, tau):
    # input
    objIptStream = ImgstreamReader(os.path.join('D:/STMD_Dataset','White-Background', 
                                                    'TW-0.8d-TH-0.8d-TV-'+str(v)+'d_s-TL-0-SamFre-200', 
                                                    'WhiteBG*.tif'))
    # FracSTMD
    objFracSTMD = instancing_model('FracSTMD')
    objFracSTMD.set_parameter(n1 = 20, tau1 = tau)
    objFracSTMD.init_config()


    maxFracSTMD = 0
    while objIptStream.hasFrame:
        # Read the next frame from the image stream
        grayImg, _ = objIptStream.get_next_frame()
        
        # Perform inference using the objModel
        resultFracSTMD = inference(objFracSTMD, grayImg)

        # record max
        if objIptStream.currIdx > 100:
            maxFracSTMD = max(maxFracSTMD, resultFracSTMD['response'].max())


        if objIptStream.currIdx == 120:
            # record to curve
            return maxFracSTMD


def process_STMDNet_task(v):
    # input
    objIptStream = ImgstreamReader(os.path.join('D:/STMD_Dataset','White-Background', 
                                                    'TW-0.8d-TH-0.8d-TV-'+str(v)+'d_s-TL-0-SamFre-200', 
                                                    'WhiteBG*.tif'))


    objSTMDNet = instancing_model('STMDNet')
    objSTMDNet.init_config()


    maxSTMDNet = 0
    while objIptStream.hasFrame:
        # Read the next frame from the image stream
        grayImg, _ = objIptStream.get_next_frame()
        
        # Perform inference using the objModel
        resultSTMDNet = inference(objSTMDNet, grayImg)

        # record max
        if objIptStream.currIdx > 100:
            maxSTMDNet = max(maxSTMDNet, resultSTMDNet['response'].max())

        if objIptStream.currIdx == 120:
            # record to curve
            return maxSTMDNet


def main():
    vList = [1, 4, 7, 8, 9] + [i*10 for i in range(1, 10)] + [i for i in range(100, 301, 25)] + [500, 1000]
    tauList = [i for i in range(1, 7, 1)] + [i for i in range(8, 22, 2)]

    curveFracSTMD = [[None for i in range(len(vList))] for _ in range(len(tauList))]
    curveSTMDNet = [None for i in range(len(vList))]

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = []
        for j, v in enumerate(vList):
            future1 = executor.submit(process_STMDNet_task, v)
            future1.i = -1
            future1.j = j   
            futures.append(future1)
            for i, tau in enumerate(tauList):
                future = executor.submit(process_Frac_task, v, tau)
                future.i = i
                future.j = j
                futures.append(future)
            
        for future in tqdm(concurrent.futures.as_completed(futures), 
                           desc='Processing Pool Executor',
                           total=len(vList)*(len(tauList)+1)):
            res = future.result()
            if future.i == -1:
                curveSTMDNet[future.j] = res
            else:
                curveFracSTMD[future.i][future.j] = res

            
    # 将列表打包到一个字典中
    data = {'curveFracSTMD': curveFracSTMD, 'curveSTMDNet': curveSTMDNet}
    # 保存到 JSON 文件
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'velocity_curve.json'), 'w') as file:
        json.dump(data, file, indent=4)

    print('\n\nDone...')


def read_json():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'velocity_curve.json'), 'r') as file:
        loaded_data = json.load(file)

    curveFracSTMD = loaded_data['curveFracSTMD']
    curveSTMDNet = loaded_data['curveSTMDNet']
    return curveFracSTMD, curveSTMDNet


def visualize():
    curveFracSTMD, curveSTMDNet = read_json()
    
    import matplotlib.pyplot as plt

    vList = [1, 4, 7, 8, 9] + [i*10 for i in range(1, 10)] + [i for i in range(100, 301, 25)] + [500, 1000]

    fig, ax = plt.subplots()
    tauList = [i for i in range(1, 7, 1)] + [i for i in range(8, 22, 2)]

    for i in range(len(curveFracSTMD)):
        if not i%4:
            curveF = curveFracSTMD[i] / np.max(curveFracSTMD[i])
            ax.plot(vList, curveF, label='tau=%d'%(tauList[i]))

    curveB = curveSTMDNet / np.max(curveSTMDNet)
    ax.plot(vList, curveB, 'r-*', linewidth=2, markeredgewidth=2, label='Proposed')
    ax.set_xscale('log')
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Velocity (degree/s)', fontsize=14)
    ax.set_ylabel('Peak Model Response (normalised)', fontsize=14)
    ax.legend()
    plt.show()
        

if __name__ == '__main__':
    # main()
    visualize()