import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from tqdm import tqdm
import torch
import numpy as np
import time


import config
from smalltargetmotiondetectors.model.vstmd import vSTMD, vSTMD_F # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader # type: ignore
from utils import custom_serialize
from RIST_config import datasetInfo, ristDatasetPath



def _task_vSTMD_cpu(input_path):

    objIptStream = VidstreamReader(input_path)

    vSTMD_cpu = vSTMD(device='cpu')
    vSTMD_cpu.init_config()
    
    total_time_spend = 0

    ''' Run '''
    i = 0
    while objIptStream.hasFrame:
        i += 1
        # Read the next frame from the video stream
        grayImg, _ = objIptStream.get_next_frame()
        grayImg = grayImg.astype(np.float32)
        
        # Perform inference using the model
        _, time_spend = vSTMD_cpu.process(grayImg)
        total_time_spend += time_spend
        

    return total_time_spend, i


def _task_vSTMD_gpu(input_path):

    objIptStream = VidstreamReader(input_path)

    vSTMD_gpu = vSTMD(device='cuda')
    vSTMD_gpu.init_config()

    total_time_spend = 0

    ''' Run '''
    i = 0
    while objIptStream.hasFrame:
        i += 1
        # Read the next frame from the video stream
        grayImg, _ = objIptStream.get_next_frame()
        gray_img_torch = torch.from_numpy(grayImg).float().unsqueeze(0).unsqueeze(0).to('cuda')
        
        # Perform inference using the model
        time_start = time.time()
        vSTMD_gpu.model_structure(gray_img_torch)
        torch.cuda.synchronize()
        time_spend = time.time() - time_start
        total_time_spend += time_spend

    return total_time_spend, i


def _task_vSTMD_F_cpu(input_path):

    objIptStream = VidstreamReader(input_path)

    vSTMD_F_cpu = vSTMD_F(device='cpu')
    vSTMD_F_cpu.init_config()


    total_time_spend = 0

    ''' Run '''
    i = 0
    while objIptStream.hasFrame:
        i += 1
        # Read the next frame from the video stream
        grayImg, _ = objIptStream.get_next_frame()
        grayImg = grayImg.astype(np.float32)
        
        # Perform inference using the model
        _, time_spend = vSTMD_F_cpu.process(grayImg)
        total_time_spend += time_spend

    return total_time_spend, i


def _task_vSTMD_F_gpu(input_path):

    objIptStream = VidstreamReader(input_path)

    vSTMD_F_gpu = vSTMD_F(device='cuda')
    vSTMD_F_gpu.init_config()

    total_time_spend = 0

    ''' Run '''
    i = 0
    while objIptStream.hasFrame:
        i += 1
        # Read the next frame from the video stream
        grayImg, _ = objIptStream.get_next_frame()
        gray_img_torch = torch.from_numpy(grayImg).float().unsqueeze(0).unsqueeze(0).to('cuda')
        
        # Perform inference using the model
        time_start = time.time()
        vSTMD_F_gpu.model_structure(gray_img_torch)
        torch.cuda.synchronize()
        time_spend = time.time() - time_start
        total_time_spend += time_spend


    return total_time_spend, i


def main_inference():

    time_spend_dict = {
        'vSTMD_cpu': 0,
        'vSTMD_F_cpu': 0,
        'vSTMD_gpu': 0,
        'vSTMD_F_gpu': 0,
    }

    frame_dict = {
        'vSTMD_cpu': 0,
        'vSTMD_F_cpu': 0,
        'vSTMD_gpu': 0,
        'vSTMD_F_gpu': 0,
    }


    
    for datasetName in tqdm(datasetInfo.keys(), desc='vSTMD_cpu'):
        # Dataset path
        inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
        time_cost, total_frame =  _task_vSTMD_cpu(inputPath)
        time_spend_dict['vSTMD_cpu'] += time_cost
        frame_dict['vSTMD_cpu'] += total_frame

    for datasetName in tqdm(datasetInfo.keys(), desc='vSTMD_gpu'):
        # Dataset path
        inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
        time_cost, total_frame =  _task_vSTMD_gpu(inputPath)
        time_spend_dict['vSTMD_gpu'] += time_cost
        frame_dict['vSTMD_gpu'] += total_frame

    for datasetName in tqdm(datasetInfo.keys(), desc='vSTMD_F_cpu'):
        # Dataset path
        inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
        time_cost, total_frame =  _task_vSTMD_F_cpu(inputPath)
        time_spend_dict['vSTMD_F_cpu'] += time_cost
        frame_dict['vSTMD_F_cpu'] += total_frame


    for datasetName in tqdm(datasetInfo.keys(), desc='vSTMD_F_gpu'):
        # Dataset path
        inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
        time_cost, total_frame =  _task_vSTMD_F_gpu(inputPath)
        time_spend_dict['vSTMD_F_gpu'] += time_cost
        frame_dict['vSTMD_F_gpu'] += total_frame


    fps_dict = {}
    for key in time_spend_dict.keys():
        time_spend_dict[key] /= frame_dict[key]  # average time cost per frame
        fps_dict[key] = 1.0 / time_spend_dict[key]

    
    with open(f'{os.path.abspath(__file__)[:-3]}.json', 'w') as f:
        json.dump({'time':time_spend_dict,
                   'FPS':fps_dict},
                    f, default=custom_serialize, indent=4)


def show_timecost():
    with open(f'{os.path.abspath(__file__)[:-3]}.json', 'r') as f:
        data = json.load(f)
    time_spend_dict = data['time']
    fps_dict = data['FPS']
    
    import prettytable

    table = prettytable.PrettyTable()
    table.field_names = ["Module"] + [key for key in time_spend_dict.keys()]
    table.add_row(["time cost"] + [f"{value:.6f}" for value in time_spend_dict.values()])
    table.add_row(["FPS"] + [f"{value:.2f}" for value in fps_dict.values()])
    print(table)

        


if __name__ == '__main__':
    # main_inference()
    show_timecost()
