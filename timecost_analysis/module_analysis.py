import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import concurrent.futures
from tqdm import tqdm
import time
import numpy as np

import config
from smalltargetmotiondetectors.model import backbone, vstmd # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader # type: ignore
from utils import custom_serialize
from RIST_config import datasetInfo, ristDatasetPath




class Delay_And_Correlate(backbone.FracSTMD):
    
    def process(self, iptMatrix):
        # Process input matrix through model components
        self.retinaOpt = self.hRetina.process(iptMatrix)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        ON_signal = np.maximum(self.laminaOpt, 0)
        OFF_signal = np.maximum(-self.laminaOpt, 0)

        time_tic = time.time()
        delay_off = self.hMedulla.hTm1.process(OFF_signal)
        out = ON_signal * delay_off
        time_spent = time.time() - time_tic

        return time_spent
    

class cIDP_location(vstmd.vSTMD):

    def __init__(self):
        super().__init__(device='cpu')

    def process(self, iptMatrix):
        # Process input matrix through model components
        self.hRetina.process(iptMatrix)
        self.hLamina.process(self.hRetina.Opt)

        time_tic = time.time()
        self.hMedulla.process(self.hLamina.Opt[0], self.hLamina.Opt[1])
        out = self.hMedulla.Opt[0] * self.hMedulla.Opt[1]
        time_spent = time.time() - time_tic

        return time_spent


class Iso_DGC(backbone.DSTMDBackbone):

    def medulla_process(self, ON_signal, OFF_signal):
        # Process signals with delays
        mi1Para4Signal = self.hMedulla.hMi1Para4.process(ON_signal)
        
        self.hMedulla.cellTm1Ipt.record_next(OFF_signal)
        tm1Para5Signal = self.hMedulla.hTm1Para5.process(self.hMedulla.cellTm1Ipt)
        tm1Para6Signal = self.hMedulla.hTm1Para6.process(self.hMedulla.cellTm1Ipt)

        # Output signals
        self.Opt = [ON_signal, mi1Para4Signal, tm1Para5Signal, tm1Para6Signal]
        return self.Opt



    def process(self, iptMatrix):
        # Process input matrix through model components
        self.retinaOpt = self.hRetina.process(iptMatrix)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        ON_signal = np.maximum(self.laminaOpt, 0)
        OFF_signal = np.maximum(-self.laminaOpt, 0)


        time_tic = time.time()
        medulla_opt = self.medulla_process(ON_signal, OFF_signal)
        self.lobulaOpt = self.hLobula.process(medulla_opt)
        time_spent = time.time() - time_tic

        return time_spent
    

class CDGC(vstmd.vSTMD):
    def __init__(self):
        super().__init__(device='cpu')

    def process(self, iptMatrix):
        # Process input matrix through model components
        self.hRetina.process(iptMatrix)
        self.hLamina.process(self.hRetina.Opt)

        time_tic = time.time()
        self.hMedulla.process(self.hLamina.Opt[0], self.hLamina.Opt[1])        
        # Process through Lobula and get response and direction
        direction = self.hLobula.hCollDireEnDecoding.process(self.hMedulla.Opt[0], 
                                                            self.hMedulla.Opt[1], 
                                                            self.hLamina.Opt[0],
                                                            self.hLamina.Opt[1])
        time_spent = time.time() - time_tic

        return time_spent


def _task(input_path):
    ''' Dynamically create a video stream reader or other input type '''


    objIptStream = VidstreamReader(input_path)

    delay_and_correlate = Delay_And_Correlate()
    delay_and_correlate.init_config()
    cIDP_loc = cIDP_location()
    cIDP_loc.init_config()
    iso_dgc = Iso_DGC()
    iso_dgc.init_config()
    cdgc = CDGC()
    cdgc.init_config()

    time_spend_dict = {
        'Delay_And_Correlate': 0,
        'cIDP_location': 0,
        'Iso_DGC': 0,
        'CDGC': 0,
    }

    ''' Run '''
    i = 0
    while objIptStream.hasFrame:
        i += 1
        # Read the next frame from the video stream
        grayImg, _ = objIptStream.get_next_frame()
        
        # Perform inference using the model

        time_spend_dict['Delay_And_Correlate'] += delay_and_correlate.process(grayImg)
        time_spend_dict['cIDP_location'] += cIDP_loc.process(grayImg)
        time_spend_dict['Iso_DGC'] += iso_dgc.process(grayImg)
        time_spend_dict['CDGC'] += cdgc.process(grayImg)

    for key in time_spend_dict.keys():
        time_spend_dict[key] /= i  # average time cost per frame

    return time_spend_dict


def main_inference():

    time_in_dataset = {}
    for datasetName in tqdm(datasetInfo.keys()):
        # Dataset path
        inputPath = os.path.join(ristDatasetPath, datasetName, f'{datasetName}.mp4')


        time_spent = _task(inputPath)

        time_in_dataset[datasetName] = time_spent


    time_spend_dict = {
        'Delay_And_Correlate': 0,
        'cIDP_location': 0,
        'Iso_DGC': 0,
        'CDGC': 0,
        }
    for value in time_in_dataset.values():
        time_spend_dict['Delay_And_Correlate'] += value['Delay_And_Correlate']
        time_spend_dict['cIDP_location'] += value['cIDP_location']
        time_spend_dict['Iso_DGC'] += value['Iso_DGC']
        time_spend_dict['CDGC'] += value['CDGC']

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
    main_inference()
    show_timecost()