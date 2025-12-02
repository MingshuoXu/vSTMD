import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import concurrent.futures
from tqdm import tqdm
import time
import numpy as np
import torch

import config
from smalltargetmotiondetectors.model.vstmd import vSTMD, vSTMD_F # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader # type: ignore
from utils import custom_serialize
from RIST_config import datasetInfo, ristDatasetPath




def _task(input_path):
    ''' Dynamically create a video stream reader or other input type '''


    objIptStream = VidstreamReader(input_path)

    vSTMD_cpu = vSTMD(device='cpu')
    vSTMD_cpu.init_config()
    vSTMD_F_cpu = vSTMD_F(device='cpu')
    vSTMD_F_cpu.init_config()
    vSTMD_gpu = vSTMD(device='cuda')
    vSTMD_gpu.init_config()
    vSTMD_F_gpu = vSTMD_F(device='cuda')
    vSTMD_F_gpu.init_config()

    time_spend_dict = {
        'vSTMD_cpu': 0,
        'vSTMD_F_cpu': 0,
        'vSTMD_gpu': 0,
        'vSTMD_F_gpu': 0,
    }

    ''' Run '''
    i = 0
    while objIptStream.hasFrame:
        i += 1
        # Read the next frame from the video stream
        grayImg, _ = objIptStream.get_next_frame()
        gray_img_torch = torch.from_numpy(grayImg).float().unsqueeze(0).unsqueeze(0).to('cuda')
        
        # Perform inference using the model
        _, time_spend = vSTMD_cpu.process(grayImg)
        time_spend_dict['vSTMD_cpu'] += time_spend
        _, time_spend = vSTMD_F_cpu.process(grayImg)
        time_spend_dict['vSTMD_F_cpu'] += time_spend
        _, time_spend = vSTMD_gpu.process(gray_img_torch)
        time_spend_dict['vSTMD_gpu'] += time_spend
        _, time_spend = vSTMD_F_gpu.process(gray_img_torch)
        time_spend_dict['vSTMD_F_gpu'] += time_spend

    for key in time_spend_dict.keys():
        time_spend_dict[key] /= i  # average time cost per frame

    return time_spend_dict


def main_inference():

    time_in_dataset = {}
    for datasetName in tqdm(datasetInfo.keys()):
        # Dataset path
        inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
        time_in_dataset[datasetName] = _task(inputPath)


    time_spend_dict = {
        'vSTMD_cpu': 0,
        'vSTMD_F_cpu': 0,
        'vSTMD_gpu': 0,
        'vSTMD_F_gpu': 0,
    }
    for value in time_in_dataset.values():
        for key in time_spend_dict.keys():
            time_spend_dict[key] += value[key]

    for key in time_spend_dict.keys():
        time_spend_dict[key] /= len(time_in_dataset)  # average time cost per frame

    
    with open(f'{os.path.abspath(__file__)[:-3]}.json', 'w') as f:
        json.dump(time_spend_dict, f, default=custom_serialize, indent=4)

def show_timecost():
    with open(f'{os.path.abspath(__file__)[:-3]}.json', 'r') as f:
        time_spend_dict = json.load(f)
    
    import prettytable

    table = prettytable.PrettyTable()
    table.field_names = ["Module"] + [key for key in time_spend_dict.keys()]
    table.add_row(["time cost"] + [f"{value:.6f}" for value in time_spend_dict.values()])
    print(table)


if __name__ == '__main__':
    # main_inference()
    show_timecost()