import os
import sys
import concurrent.futures

import numpy as np
from tqdm import tqdm
import json


ProjectPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ProjectPath)
import config
from utils import custom_serialize
from smalltargetmotiondetectors.api import instancing_model, inference # type: ignore
from smalltargetmotiondetectors.util.iostream import ImgstreamReader # type: ignore

apiOpticFlowPth = os.path.join(ProjectPath, 'comparison_models')
sys.path.append(apiOpticFlowPth)
from custom_API import (CustomFlowDiffuser, CustomRAFT, CustomSEA_RAFT,  # type: ignore
                        CustomMemFlow, CustomStreamFlow, CustomDpFlow, 
                        img2tensor, flow_to_ang) 

directionalStmdList = ('DSTMD', 'STMDPlus', 'ApgSTMD', 'vSTMD', 'vSTMD_F') 
opticflowModelList = ('RAFT', 'SEA_RAFT',
                      'MemFlow', 'StreamFlow', 'DpFlow', 'FlowDiffuser') 
V_LIST = [v for v in range(100, 3001, 100)]
TIME_END = 500


# Define paths
def get_input_path(velocity):
    inputPath = os.path.join('D:/', 'STMD_Dataset', 'vSTMD_Panorama_Stimuli', 'Bgr_dire=Leftward_v=250',
                             f'ET-Target_Num=1_W=5_H=5_V={velocity}_L=0-Traj=Ellipse_FPS=1000',
                             'vSTMD_Panorama_Stimuli*.tif')
    return inputPath


def get_inference_output_path(model_name, velocity):
    outputPath = os.path.join('D:/', 'STMD_Dataset', 'inference_vSTMD_Panorama_Stimuli', 'Bgr_dire=Leftward_v=250',
                              f'ET-Target_Num=1_W=5_H=5_V={velocity}_L=0-Traj=Ellipse_FPS=1000',
                             f'{model_name}_direction.json')
    return outputPath


def get_evaluate_output_path(velocity):
    outputPath = os.path.join(ProjectPath, 'evaluate_result', 'vSTMD_Panorama_Stimuli',
                              f'ET-Target_Num=1_W=5_H=5_V={velocity}_L=0-Traj=Ellipse_FPS=1000.json')
    return outputPath


# groundtruth reading
def read_groundtruth(velocity, timeEnd):
    with open(os.path.join(ProjectPath, 'groundtruth', 'Bgr_dire=Leftward_v=250',
                           f'ET-Target_Num=1_W=5_H=5_V={velocity}_L=0-Traj=Ellipse_FPS=1000.json'),
                             'r') as f:
        data = json.load(f)

    # groundtruth
    posiGT = np.zeros((timeEnd,2))
    DireGT = np.zeros(timeEnd)
    for tt in range(0, timeEnd):
        GT = data[tt]['bbox']
        posiGT[tt,:] = (GT[0]+3, GT[1]+3)
        if tt > 0:
            DireGT[tt] = data[tt]['direction']
        else:
            DireGT[tt] = None  # first frame use the second frame direction

    return posiGT, DireGT


# update evaluate result json
def _update_direction_evaluate_result_json(velocity, model_name, key, value):
    json_file_name = get_evaluate_output_path(velocity)

    if os.path.exists(json_file_name):
        with open(json_file_name, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[model_name][key] = value
    data = custom_serialize(data, indent=2)

    with open(json_file_name, 'w') as f:
        f.write(data)


# inference functions
def inferece_STMD_direction(model_name, model, velocity, timeEnd):
    # record the direction

    def find_direction_within_radius(result, x0, y0, radius):
        # get the response matrix and direction matrix
        response = result['response']
        direction = result['direction']

        # get the shape of the response matrix
        height, width = response.shape

        # create a mask to get the region of interest
        x_indices, y_indices = np.ogrid[0:height, 0:width]
        mask = (x_indices - x0) ** 2 + (y_indices - y0) ** 2 <= radius ** 2

        # get the response matrix within the region
        region_response = np.where(mask, response, 0)

        # get the maximum value within the region
        max_value = np.nanmax(region_response)
        if max_value > 0:
            xD, yD = np.unravel_index(np.nanargmax(region_response), region_response.shape)
            return direction[xD, yD]  # return the direction of the maximum value
        else:
            return np.nan  # return NaN if no response is found
            
    ''' Input '''
    hSteam = ImgstreamReader(get_input_path(velocity))

    posiGT, DireGT = read_groundtruth(velocity, timeEnd)

    direction_list = [None for _ in range(timeEnd)]

    # Run inference
    for countT in range(timeEnd):
        # Get the next frame from the input source
        grayImg, _ = hSteam.get_next_frame()
        
        # Perform inference using the model
        resultvSTMD, _ = inference(model, grayImg)
        
        # 对两个结果集调用该函数
        x0, y0 = posiGT[countT,:]
        direction_list[countT] = find_direction_within_radius(resultvSTMD, x0, y0, 8)
        
    with open( get_inference_output_path(model_name, velocity), 'w') as f:
        json.dump({'direction': direction_list}, f)


def inference_OF_models_direction(model_name, model, velocity, timeEnd):

    def find_direction_within_radius(flow, direGT, x0, y0, radius):

        angMtx = flow_to_ang(flow)  # Convert flow to angle matrix if needed
        x1 = max(0, x0-radius)  # Ensure x is not negative
        x2 = min(angMtx.shape[1], x0+radius)  # Ensure x does not exceed the width
        y1 = max(0, y0-radius)  # Ensure y is not negative
        y2 = min(angMtx.shape[0], y0+radius)
        region = angMtx[y1:y2, x1:x2]  # Extract the region of interest

        return region.flatten().tolist()
    
    ''' Input '''
    hSteam = ImgstreamReader(
        os.path.join('D:\\', 'STMD_Dataset', 'vSTMD_Panorama_Stimuli', 'Bgr_dire=Leftward_v=250',
                    f'ET-Target_Num=1_W=5_H=5_V={velocity}_L=0-Traj=Ellipse_FPS=1000',
                    'vSTMD_Panorama_Stimuli*.tif')
                    )

    posiGT, DireGT = read_groundtruth(velocity, timeEnd)
    
    direction_list = []

    # Run inference
    for count in range(timeEnd):
        # Get the next frame from the input source
        
        _, colorImg = hSteam.get_next_frame()

        colorImg = np.pad(colorImg, ((0, 2), (0, 2), (0, 0)), mode='constant', constant_values=0)
        # Perform inference using the model        
        frame = img2tensor(colorImg)

        flow = model.process(newFrame=frame)


        # postprocessing
        if flow is not None:            
            if isinstance(flow, list):
                for j, f in enumerate(flow):
                    k = len(flow) - 1 - j
                    x, y = posiGT[count-k]
                    direGT = DireGT[count-k]
                    if j < len(flow)-1:
                        # j=1 -> idx=-1; j=0 -> idx=-2; 
                        direction_list[-k] = find_direction_within_radius(f, direGT, int(x), int(y), 8)
                    else:
                        # j=2 -> append
                        direction_list.append(find_direction_within_radius(f, direGT, int(x), int(y), 8))
            else:
                x, y = posiGT[count]
                direGT = DireGT[count]
                direction_list.append(find_direction_within_radius(flow, direGT, int(x), int(y), 8))
        else:
            direction_list.append(np.nan)
        
    with open( get_inference_output_path(model_name, velocity), 'w') as f:
        json.dump({'direction': direction_list}, f)


def _inference_STMD_task(model_name, velocity, timeEnd):
    
    ''' Initialize the model '''
    model = instancing_model(model_name)

    # set the parameter list
    tau = round(5 / (velocity / 1000))

    if model_name == 'DSTMD':
        model.set_parameter( n4 = round(tau*0.3), tau4 = round(tau*0.6), 
                                n5 = round(tau*0.5), tau5 = tau, 
                                n6 = round(tau*0.8), tau6 = round(tau*1.6)
                                )  
    elif model_name == 'STMDPlus':
        model.set_parameter( n3 = round(tau*0.3), tau3 = round(tau*0.6), 
                                n4 = round(tau*0.5), tau4 = tau, 
                                n5 = round(tau*0.8), tau5 = round(tau*1.6)
                                )  
    elif model_name == 'ApgSTMD':
        model.set_parameter( n3 = round(tau*0.3), tau3 = round(tau*0.6), 
                                n4 = round(tau*0.5), tau4 = tau, 
                                n5 = round(tau*0.8), tau5 = round(tau*1.6)
                                )  
           

    # init
    model.init_config()

    inferece_STMD_direction(model_name, model, velocity, timeEnd)
    

def _inference_OF_task(model_name, velocity, timeEnd):
    ''' Initialize the model '''
    if model_name == 'RAFT':
        model = CustomRAFT()
    elif model_name == 'SEA_RAFT':
        model = CustomSEA_RAFT()
    elif model_name == 'MemFlow':
        model = CustomMemFlow()
    elif model_name == 'StreamFlow':
        model = CustomStreamFlow()
    elif model_name == 'DpFlow':
        model = CustomDpFlow()
    elif model_name == 'FlowDiffuser':
        model = CustomFlowDiffuser()

    inference_OF_models_direction(model_name, model, velocity, timeEnd)
    

def main_inference(max_workers=6):
    time_end = TIME_END
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for v in V_LIST:
            for model_name in directionalStmdList:
                futures.append(executor.submit(_inference_STMD_task, model_name, v, time_end))
        
            for model_name in opticflowModelList:
                futures.append(executor.submit(_inference_OF_task, model_name, v, time_end))

        for future in tqdm(concurrent.futures.as_completed(futures), 
                           total=len(futures), 
                           desc='inference direction'):
            future.result()


# evaluation functions
def _evaluate_STMD_task(model_name, velocity, timeEnd):
    # read inference result
    with open(get_inference_output_path(model_name, velocity), 'r') as f:
        data = json.load(f)
    direction_list = np.array(data['direction'])

    # read groundtruth
    _, DireGT = read_groundtruth(velocity, timeEnd)

    # compute err
    err = abs(direction_list - DireGT)
    err[err > np.pi] = 2 * np.pi - err[err > np.pi]  # ensure the error is within [0, pi]
    
    AAE = np.nanmean(err[:timeEnd])

    _update_direction_evaluate_result_json(velocity, model_name, 'AAR', AAE)


def _evaluate_OF_task(model_name, velocity, timeEnd):
    # read inference result
    with open(get_inference_output_path(model_name, velocity), 'r') as f:
        data = json.load(f)
    direction_list = np.array(data['direction'])

    # read groundtruth
    _, direction_GT = read_groundtruth(velocity, timeEnd)

    direction_err_list = []
    for directions in direction_list:
        dire_err_list = abs(directions - direction_GT)  # Calculate the error
        dire_err_list[dire_err_list > np.pi] = 2 * np.pi - dire_err_list[dire_err_list > np.pi]  # ensure the error is within [0, pi]
        sort_err_list = np.sort(dire_err_list)
        AE = np.mean(sort_err_list[:int(len(sort_err_list)/2)]) 
        direction_err_list.append(AE)

    AAE = np.nanmean(direction_err_list[:timeEnd])

    _update_direction_evaluate_result_json(velocity, model_name, 'AAR', AAE)
    

def main_evaluate(max_workers=6):
    for model_name in directionalStmdList:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for v in V_LIST:
                futures.append(executor.submit(_evaluate_STMD_task, model_name, v, TIME_END))

            for future in tqdm(concurrent.futures.as_completed(futures), 
                            total=len(futures), 
                            desc='inference direction'):
                future.result()


    for model_name in opticflowModelList:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for v in V_LIST:
                futures.append(executor.submit(_evaluate_OF_task, model_name, v, TIME_END))

            for future in tqdm(concurrent.futures.as_completed(futures), 
                            total=len(futures), 
                            desc='inference direction'):
                future.result()


# collect results
def collect_results():
    jsonFileName = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'direction_error_across_velocity.json')

    errDict = {name: [0 for _ in range(len(V_LIST))] for name in directionalStmdList+opticflowModelList}

    for i, v in enumerate(V_LIST):
        with open(get_evaluate_output_path(v), 'r') as f:
            data = json.load(f)
        for model_name in directionalStmdList+opticflowModelList:
            errDict[model_name][i] = data[model_name]['AAR']

    with open(jsonFileName, 'w') as f:
        json.dump({'velocity': list(V_LIST), 'errDict': errDict, }, f)


if __name__ == '__main__':
    main_inference(8)
    # main_evaluate()
    # collect_results()