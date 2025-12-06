import os
import sys
ITEM_PTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ITEM_PTH)
import time

import json
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

import config
from smalltargetmotiondetectors.model import backbone, vstmd # type: ignore
from smalltargetmotiondetectors.util.iostream import ImgstreamReader # type: ignore
from utils import custom_serialize


def get_input_stream(velocity):
    ''' Dynamically create a video stream reader or other input type '''
    input_template = os.path.join('D:/', 'STMD_Dataset', 'vSTMD_Panorama_Stimuli', 'White-Background',
                             f'TW-0.8d-TH-0.8d-TV-{velocity}_pixel_s-TL-0-SamFre-1000',
                             'WhiteBG*.tif')
    objIptStream = ImgstreamReader(input_template)

    return objIptStream


class Delay_And_Correlate(backbone.FracSTMD):
    
    def process(self, iptMatrix):
        # Process input matrix through model components
        self.retinaOpt = self.hRetina.process(iptMatrix)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        self.on_signal = np.maximum(self.laminaOpt, 0)
        self.off_signal = np.maximum(-self.laminaOpt, 0)


        self.delay_off = self.hMedulla.hTm1.process(self.on_signal)
        self.out = self.on_signal * self.delay_off

    
    

class cIDP_location(vstmd.vSTMD):

    def __init__(self):
        super().__init__(device='cpu')

    def process(self, iptMatrix):
        # Process input matrix through model components
        self.hRetina.process(iptMatrix)
        self.hLamina.process(self.hRetina.Opt)

        self.L_on = self.hLamina.Opt[0]
        self.L_off = self.hLamina.Opt[1]
        self.hMedulla.process(self.L_on, self.L_off)
        self.v_on = self.hMedulla.Opt[0]
        self.v_off = self.hMedulla.Opt[1]
        self.out = self.v_on * self.v_off





def _task(v):
    ''' Dynamically create a video stream reader or other input type '''

    input_stream = get_input_stream(v)

    delay_and_correlate = Delay_And_Correlate()
    delay_and_correlate.init_config()
    cIDP_loc = cIDP_location()
    cIDP_loc.init_config()


    delay_and_correlate_output = {
        'L_on': [],
        'L_off': [],
        'delay_off': [],
        'out': []
    }
    cIDP_loc_output = {
        'L_on': [],
        'L_off': [],
        'v_on': [],
        'v_off': [],
        'out': []
    }

    x = int(310/2)
    y = int(470-200)

    ''' Run '''

    for i in tqdm(range(500)):
        # Read the next frame from the video stream
        grayImg, _ = input_stream.get_next_frame()
        
        # Perform inference using the model
        delay_and_correlate.process(grayImg)
        delay_and_correlate_output['L_on'].append(delay_and_correlate.on_signal[x, y])
        delay_and_correlate_output['L_off'].append(delay_and_correlate.off_signal[x, y])
        delay_and_correlate_output['delay_off'].append(delay_and_correlate.delay_off[x, y])
        delay_and_correlate_output['out'].append(delay_and_correlate.out[x, y])


        cIDP_loc.process(grayImg)
        cIDP_loc_output['L_on'].append(cIDP_loc.L_on[x, y])
        cIDP_loc_output['L_off'].append(cIDP_loc.L_off[x, y])
        cIDP_loc_output['v_on'].append(cIDP_loc.v_on[x, y])
        cIDP_loc_output['v_off'].append(cIDP_loc.v_off[x, y])
        cIDP_loc_output['out'].append(cIDP_loc.out[x, y])


    return delay_and_correlate_output, cIDP_loc_output


def collect_data():
    v_list = [500, 1000, 2000, 3000]

    delay_and_correlate_results = {}
    cIDP_loc_results = {}
    for v in v_list:
        delay_and_correlate_output, cIDP_loc_output = _task(v)
        delay_and_correlate_results[f'{v}'] = delay_and_correlate_output
        cIDP_loc_results[f'{v}'] = cIDP_loc_output

    with open(os.path.join(ITEM_PTH, 'new_correlation_modelling', 'delay_and_correlate_results.json'), 'w') as f:
        data = {'delay_and_correlate_results': delay_and_correlate_results,
                'cIDP_loc_results': cIDP_loc_results}
        data = custom_serialize(data)
        f.write(data)


def show_results():
    with open(os.path.join(ITEM_PTH, 'new_correlation_modelling', 'delay_and_correlate_results.json'), 'r') as f:
        data = json.load(f)
    
    delay_and_correlate_results = data['delay_and_correlate_results']
    cIDP_loc_results = data['cIDP_loc_results']

    fig, axs = plt.subplots(4, 2, figsize=(12, 10))
    v_list = [500, 1000, 2000, 3000]
    for i, v in enumerate(v_list):
        ax_left = axs[i, 0]
        ax_left.plot(delay_and_correlate_results[f'{v}']['out'], label='Delay and Correlate Output')
        ax_left.plot(delay_and_correlate_results[f'{v}']['delay_off'], label='Delay Off Signal', linestyle='--')
        ax_left.plot(delay_and_correlate_results[f'{v}']['L_on'], label='L On Signal', linestyle=':')
        ax_left.plot(delay_and_correlate_results[f'{v}']['L_off'], label='L Off Signal', linestyle='-.')
        ax_left.set_title(f'Delay and Correlate Model Output at {v} pixel/s')
        ax_left.set_xlabel('Frame')
        ax_left.set_ylabel('Response')
        ax_left.legend()

        ax_right = axs[i, 1]
        ax_right.plot(cIDP_loc_results[f'{v}']['out'], label='cIDP Location Output', color='orange')
        ax_right.plot(cIDP_loc_results[f'{v}']['v_on'], label='v On Signal', linestyle='--', color='green')
        ax_right.plot(cIDP_loc_results[f'{v}']['v_off'], label='v Off Signal', linestyle=':', color='red')
        ax_right.plot(cIDP_loc_results[f'{v}']['L_on'], label='L On Signal', linestyle='-.', color='purple')
        ax_right.plot(cIDP_loc_results[f'{v}']['L_off'], label='L Off Signal', linestyle='-', color='brown')
        ax_right.set_title(f'cIDP Location Model Output at {v} pixel/s')
        ax_right.set_xlabel('Frame')
        ax_right.set_ylabel('Response')
        ax_right.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # collect_data()
    show_results()