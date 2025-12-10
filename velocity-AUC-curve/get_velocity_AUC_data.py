import os
import sys
import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm

# Add the path to the package containing the models
TOP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # code top path
sys.path.append(TOP_PATH)
import config
from smalltargetmotiondetectors.api import inference_task, evaluate_task # type: ignore
from utils import custom_serialize


V_LIST = list(range(100, 1000, 100)) + list(range(1000, 2000, 200)) + list(range(2000, 10001, 500))
TAU_LIST = [1, 5, 15, 25, 35]

IPT_PATH = os.path.join('D:/', 'STMD_Dataset', 'vSTMD_Panorama_Stimuli', ) # input path
GROUNDTRUTH_PATH = os.path.join(TOP_PATH, 'groundtruth') # groundtruth path
EVA_OPT_PATH = os.path.join('D:/', 'STMD_Dataset', 'inference_vSTMD_Panorama_Stimuli') # evaluation path
RESULT_PATH = os.path.join(TOP_PATH, 'evaluate_result') # result path
FINAL_RES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'velocity_AUC_curve.json') # final result path


def save_json(data, filename, indent=4):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    data = custom_serialize(data, indent=indent)
    with open(filename, 'w') as file:
        file.write(data)


def get_input_path(v):
    inputpath = os.path.join(IPT_PATH, 'Bgr_dire=Leftward_v=250',
        f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS=1000',
        'vSTMD_Panorama_Stimuli*.tif')
    return inputpath


def get_inference_opt_path(modelName, v, tau=None):
    if tau is not None:
        return os.path.join(EVA_OPT_PATH, 'Bgr_dire=Leftward_v=250',
                            f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS=1000',
                            f'{modelName}_tau={tau}_opt.json')
    else:
        return os.path.join(EVA_OPT_PATH, 'Bgr_dire=Leftward_v=250',
                            f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS=1000',
                            f'{modelName}_without_tau_opt.json')


def get_evaluation_result_path(v):
    return os.path.join(RESULT_PATH, 'vSTMD_Panorama_Stimuli', 'Bgr_dire=Leftward_v=250',
                            f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS=1000.json')
    
    
def update_evaluated_result_to_json(modelName, v, tau, update_dict):
    save_path = get_evaluation_result_path(v)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    if os.path.exists(save_path):
        with open(save_path, 'r') as file:
            data = json.load(file)
    else:
        data = {}

    save_key = f'{modelName}_tau={tau}' if tau is not None else f'{modelName}_without_tau'
    if save_key in data.keys():
        for key, value in update_dict.items():
            data[save_key][key] = value
    else:
        data[save_key] = update_dict
    
    data = custom_serialize(data)
    with open(save_path, 'w') as f:
        f.write(data)


def ESTMD_task(v, tau):
    modelName = 'ESTMD'

    inputpath = get_input_path(v)

    '''inference'''
    if tau == 1:
        modelOpt, _, totalRunningTime = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500, 
                                                n1 = 1, tau1 = 1,
                                                n2 = 1, tau2 = 2,
                                                n3 = 1, tau3 = 1
                                            )
    elif tau == 5 or tau == 3:
        modelOpt, _, totalRunningTime = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500, 
                                                n1 = 1, tau1 = 2,
                                                n2 = 2, tau2 = 4,
                                                n3 = round(tau*0.5), tau3 = tau
                                            )
    else:
        modelOpt, _, totalRunningTime,  = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500, 
                                                n3 = round(tau*0.5), tau3 = tau
                                            )

    # save
    save_json(  {'modelOpt': modelOpt, 'modelDire': None, 'runningtime': totalRunningTime},
                get_inference_opt_path(modelName, v, tau), indent=2)


def DSTMD_task(v, tau):
    modelName = 'DSTMD'

    inputpath = get_input_path(v)

    '''inference'''
    if tau == 1:
        modelOpt, modelDire, totalRunningTime = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500, 
                                                n1 = 1, tau1 = 1,
                                                n2 = 2, tau2 = 2,
                                                n4 = 1, tau4 = 1, n5 = 1, tau5 = 1, n6 = 1, tau6 = 2
                                                )
    elif tau == 5 or tau == 3:
        modelOpt, modelDire, totalRunningTime = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500, 
                                                n1 = 1, tau1 = 2,
                                                n2 = 2, tau2 = 4,
                                                n4 = round(tau*0.3), tau4 = round(tau*0.6), 
                                                n5 = round(tau*0.5), tau5 = tau, 
                                                n6 = round(tau*0.8), tau6 = round(tau*1.6)
                                                )
    else:
        modelOpt, modelDire, totalRunningTime = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500, 
                                                n4 = round(tau*0.3), tau4 = round(tau*0.6), 
                                                n5 = round(tau*0.5), tau5 = tau, 
                                                n6 = round(tau*0.8), tau6 = round(tau*1.6)
                                                )
    # save
    save_json(  {'modelOpt': modelOpt, 'modelDire': modelDire, 'runningtime': totalRunningTime},
                get_inference_opt_path(modelName, v, tau), indent=2)


def FeedbackSTMD_task(v, tau):
    modelName = 'FeedbackSTMD'

    inputpath = get_input_path(v)

    '''inference'''
    if tau == 1:
        modelOpt, _, totalRunningTime = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500, 
                                                n1 = 1, tau1 = 1,
                                                n2 = 1, tau2 = 2,
                                                n3 = 1, tau3 = 1
                                            )
    elif tau == 5 or tau == 3:
        modelOpt, _, totalRunningTime = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500, 
                                                n1 = 1, tau1 = 2,
                                                n2 = 2, tau2 = 4,
                                                n3 = round(tau*0.5), tau3 = tau
                                            )
    else:
        modelOpt, _, totalRunningTime,  = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500, 
                                                n3 = round(tau*0.5), tau3 = tau
                                            )

    # save
    save_json(  {'modelOpt': modelOpt, 'modelDire': None, 'runningtime': totalRunningTime},
                get_inference_opt_path(modelName, v, tau), indent=2)


def vSTMD_task(v):
    modelName = 'vSTMD'

    inputpath = get_input_path(v)

    '''inference'''
    modelOpt, modelDire, totalRunningTime = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500,
                                                           alpha = 0.8,
                                                           device='cuda')
    
    # save
    save_json(  {'modelOpt': modelOpt, 'modelDire': modelDire, 'runningtime': totalRunningTime},
                get_inference_opt_path(modelName, v, None), indent=2)
    

def vSTMD_F_task(v):
    modelName = 'vSTMD_F'

    inputpath = get_input_path(v)

    '''inference'''
    modelOpt, modelDire, totalRunningTime = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500,
                                                           alpha = 0.8,
                                                           device='cuda')
    
    # save
    save_json(  {'modelOpt': modelOpt, 'modelDire': modelDire, 'runningtime': totalRunningTime},
                get_inference_opt_path(modelName, v, None), indent=2)


def main_inference(max_workers = 8):
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for v in V_LIST:
            for tau in TAU_LIST:
                futures.append(executor.submit(ESTMD_task, v, tau))
                futures.append(executor.submit(DSTMD_task, v, tau))
                futures.append(executor.submit(FeedbackSTMD_task, v, tau))
            futures.append(executor.submit(vSTMD_task, v))
            futures.append(executor.submit(vSTMD_F_task, v))


        for future in tqdm(concurrent.futures.as_completed(futures), 
                           desc='inference task',
                           total=len(futures)
                           ):
            future.result()


def get_groundtruth(v):
    with open(os.path.join(GROUNDTRUTH_PATH, 'Bgr_dire=Leftward_v=250',
                        f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS=1000.json'),
        'r') as file:
        gt_data = json.load(file)
    bboxs = []
    for data in gt_data:
        bboxs.append( [data['bbox'],] )
    return bboxs


def custom_evaluation(modelName, v, tau=None):
    # load modelOpt
    with open(get_inference_opt_path(modelName, v, tau), 'r') as file:
        data = json.load(file)
    modelOpt = data['modelOpt']

    # load groundtruth
    
    groundTruth = get_groundtruth(v)

    # evaluate
    AUC, AR, AP = evaluate_task(modelOpt, groundTruth, gTError=3, startFrame=80, endFrame=480, plotFigures=False)

    # save
    update_evaluated_result_to_json(modelName, v, tau, {'AUC': AUC, 'AR': AR, 'AP': AP})


def main_evaluation(max_workers = 12):
    modelNameList = ['ESTMD', 'DSTMD', 'FeedbackSTMD']
    for model_name in modelNameList:
        for tau in TAU_LIST:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for v in V_LIST:
                    futures.append(executor.submit(custom_evaluation, model_name, v, tau))

                for future in tqdm(concurrent.futures.as_completed(futures), 
                                desc=f'evaluate {model_name}_tau={tau}',
                                total=len(futures)
                                ):
                    future.result()


    modelNameList = ['vSTMD', 'vSTMD_F']
    for model_name in modelNameList:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for v in V_LIST:
                futures.append(executor.submit(custom_evaluation, model_name, v))

            for future in tqdm(concurrent.futures.as_completed(futures), 
                            desc=f'evaluate {model_name}',
                            total=len(futures)
                            ):
                future.result()
    
    
def collect_results():
    
    AUC_curve_ESTMD = [[None for j in range(len(V_LIST))] for _ in range(len(TAU_LIST))]
    AUC_curve_DSTMD = [[None for j in range(len(V_LIST))]  for _ in range(len(TAU_LIST))]
    AUC_curve_FeedbackSTMD = [[None for j in range(len(V_LIST))]  for _ in range(len(TAU_LIST))]
    AUC_curve_vSTMD = [None for j in range(len(V_LIST))]
    AUC_curve_vSTMD_F = [None for j in range(len(V_LIST))]


    for j, v in enumerate(V_LIST):
        with open(get_evaluation_result_path(v), 'r') as file:
            data = json.load(file)

        for i, tau in enumerate(TAU_LIST):
            AUC_curve_ESTMD[i][j] = data[f'ESTMD_tau={tau}']['AUC']
            AUC_curve_DSTMD[i][j] = data[f'DSTMD_tau={tau}']['AUC']
            AUC_curve_FeedbackSTMD[i][j] = data[f'FeedbackSTMD_tau={tau}']['AUC']


        AUC_curve_vSTMD[j] = data['vSTMD_without_tau']['AUC']
        AUC_curve_vSTMD_F[j] = data['vSTMD_F_without_tau']['AUC']

    
    # 将列表打包到一个字典中
    data = {'auc_curve_ESTMD': AUC_curve_ESTMD,  
            'auc_curve_DSTMD': AUC_curve_DSTMD, 
            'auc_curve_FeedbackSTMD': AUC_curve_FeedbackSTMD, 
            'auc_curve_vSTMD': AUC_curve_vSTMD, 
            'auc_curve_vSTMD_F': AUC_curve_vSTMD_F,
            'v_list': V_LIST,
            'tau_list': TAU_LIST
            }

    # 保存到 JSON 文件
    data = custom_serialize(data)
    with open(FINAL_RES_PATH, 'w') as file:
        file.write(data)

    print('\n\nDone...')

    

if __name__ == '__main__':
    main_inference(12)
    main_evaluation(12)
    collect_results()








