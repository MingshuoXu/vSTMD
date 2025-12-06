import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from tqdm import tqdm
import torch
import numpy as np
import prettytable as pt


import config
from smalltargetmotiondetectors.model.vstmd import vSTMD, vSTMD_F # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader # type: ignore
from utils import custom_serialize
from RIST_config import datasetInfo, ristDatasetPath



class vSTMD_multi_process():
    def __init__(self):
        self.vSTMD_cpu = vSTMD(device='cpu')
        self.vSTMD_gpu = vSTMD(device='cuda')
        
    def init_config(self):
        self.vSTMD_cpu.init_config()
        self.vSTMD_gpu.init_config()

    def forward_cpu(self, input_matirx):
        return self.vSTMD_cpu.process(input_matirx)
    
    def forward_gpu(self, input_matirx):
        input_matirx = torch.from_numpy(input_matirx).float().unsqueeze(0).unsqueeze(0).to('cuda')
        return self.vSTMD_gpu.process(input_matirx)

    def forward(self, input_matirx):
        self.res_cpu, _ = self.forward_cpu(input_matirx)
        self.res_gpu, _ = self.forward_gpu(input_matirx)
        torch.cuda.synchronize()
        if self.analysis_diff(self.res_cpu['response'], self.res_gpu['response'])[0] > 1e-8:
            print(f'response diff: {self.analysis_diff(self.res_cpu['response'], self.res_gpu['response'])}')




    @staticmethod
    def analysis_diff(cpu_res=None, gpu_res=None):

        diff = np.abs(cpu_res - gpu_res.squeeze(0).squeeze(0).cpu().numpy())
        max_diff = np.max(diff)

        ab_diff = np.zeros_like(cpu_res)
        np.divide(max_diff, cpu_res, 
                         out=ab_diff, 
                         where=cpu_res>0)
        max_ab_diff = np.max(ab_diff)
        
        return max_diff, max_ab_diff
    
    @staticmethod
    def direction_diff(cpu_res=None, gpu_res=None):

        diff = np.minimum(
            np.abs(cpu_res - gpu_res.squeeze(0).squeeze(0).cpu().numpy()),
            np.pi*2 - np.abs(cpu_res - gpu_res.squeeze(0).squeeze(0).cpu().numpy())
        )
        max_diff = np.max(diff)
        
        return max_diff

      

def _task(input_path):

    objIptStream = VidstreamReader(input_path)

    model = vSTMD_multi_process()
    model.init_config()
    

    ''' Run '''
    while objIptStream.hasFrame:
        # Read the next frame from the video stream
        grayImg, _ = objIptStream.get_next_frame()
        grayImg = grayImg.astype(np.float32)
        
        # Perform inference using the model
        model.forward(grayImg)


def main():

    for datasetName in tqdm(datasetInfo.keys(), desc='vSTMD_F_gpu'):
        # Dataset path
        inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')
        _task(inputPath)

        

if __name__ == '__main__':
    main()

