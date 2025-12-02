import os
import sys

import json

import concurrent.futures
from tqdm import tqdm

# Add the path to the package containing the models
TOP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # code top path
sys.path.append(TOP_PATH)
import config
from smalltargetmotiondetectors.api import inference_task, evaluate_task # type: ignore
from utils import custom_serialize

# Configuration
FPS = 1000
TOTAL_FRAME = 500

V_LIST = [_ for _ in range(100, 3001, 100)]
TAU_LIST = [1,9,17,25,33]

PANORAMA_STIMULI_FOLDER = os.path.join('D:/', 'STMD_Dataset', 'vSTMD_Panorama_Stimuli') # input path
GROUNDTRUTH_FOLDER = os.path.join(TOP_PATH, 'groundtruth') # groundtruth path
INFERENCE_OPT_FOLDER = os.path.join('D:/', 'STMD_Dataset', 'inference_vSTMD_Panorama_Stimuli', ) # evaluation path
EVALUATED_RESULT_FOLDER = os.path.join(TOP_PATH, 'evaluate_result', 'vSTMD_Panorama_Stimuli') # result path
BGR_TYPE = 'Bgr_dire=Leftward_v=250'

modelNameList = ['ESTMD', 'DSTMD', 'FracSTMD', 'vSTMD', 'vSTMD_F']


def save_json(data, filename, indent=4):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    data = custom_serialize(data, indent=indent)
    with open(filename, 'w') as file:
        file.write(data)


def update_json(model_name, data, filename, indent=4):
    if not os.path.exists(filename):
        existing_data = {}
    else:
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    
    existing_data[model_name] = data

    save_json(existing_data, filename, indent)


def _get_inputpath(v):
    inputpath = os.path.join(PANORAMA_STIMULI_FOLDER, BGR_TYPE,
        f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS={FPS}',
        'vSTMD_Panorama_Stimuli*.tif')
    return inputpath


def _get_inference_opt_path(modelName, v, tau=None):
    if tau is not None:
        return os.path.join(INFERENCE_OPT_FOLDER, BGR_TYPE,
                            f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS={FPS}',
                            f'{modelName}_tau={tau}_opt.json')
    else:
        return os.path.join(INFERENCE_OPT_FOLDER, BGR_TYPE,
                            f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS={FPS}',
                            f'{modelName}_without_tau_opt.json')


def _get_save_result_path(v):
    return os.path.join(EVALUATED_RESULT_FOLDER, BGR_TYPE,
                        f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS={FPS}.json')


def _ESTMD_task(v, tau):
    modelName = 'ESTMD'

    inputpath = _get_inputpath(v)

    '''inference'''
    if tau == 1:
        modelOpt, _, _ = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=TOTAL_FRAME, 
                                                n1 = 1, tau1 = 1,
                                                n2 = 1, tau2 = 2,
                                                n3 = 1, tau3 = 1
                                            )
    elif tau == 5 or tau == 3:
        modelOpt, _, _ = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=TOTAL_FRAME, 
                                                n1 = 1, tau1 = 2,
                                                n2 = 2, tau2 = 4,
                                                n3 = round(tau*0.5), tau3 = tau
                                            )
    else:
        modelOpt, _, _,  = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=TOTAL_FRAME, 
                                                n3 = round(tau*0.5), tau3 = tau
                                            )

    # save
    save_json(  {'modelOpt': modelOpt, 'modelDire': None, 'runningtime': None},
                _get_inference_opt_path(modelName, v, tau),
                indent=2)


def _DSTMD_task(v, tau):
    modelName = 'DSTMD'

    inputpath = _get_inputpath(v)

    '''inference'''
    if tau == 1:
        modelOpt, modelDire, _ = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=TOTAL_FRAME, 
                                                n1 = 1, tau1 = 1,
                                                n2 = 2, tau2 = 2,
                                                n4 = 1, tau4 = 1, n5 = 1, tau5 = 1, n6 = 1, tau6 = 2
                                                )
    elif tau == 5 or tau == 3:
        modelOpt, modelDire, _ = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=TOTAL_FRAME, 
                                                n1 = 1, tau1 = 2,
                                                n2 = 2, tau2 = 4,
                                                n4 = round(tau*0.3), tau4 = round(tau*0.6), 
                                                n5 = round(tau*0.5), tau5 = tau, 
                                                n6 = round(tau*0.8), tau6 = round(tau*1.6)
                                                )
    else:
        modelOpt, modelDire, _ = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=TOTAL_FRAME, 
                                                n4 = round(tau*0.3), tau4 = round(tau*0.6), 
                                                n5 = round(tau*0.5), tau5 = tau, 
                                                n6 = round(tau*0.8), tau6 = round(tau*1.6)
                                                )
    # save
    save_json(  {'modelOpt': modelOpt, 'modelDire': modelDire},
                _get_inference_opt_path(modelName, v, tau),
                indent=2)


def _FracSTMD_task(v, tau):
    modelName = 'FracSTMD'

    inputpath = _get_inputpath(v)

    '''inference'''
    modelOpt, _, _ = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=TOTAL_FRAME, 
                                    tau1 = tau)

    # save
    save_json(  {'modelOpt': modelOpt, 'modelDire': None},
                _get_inference_opt_path(modelName, v, tau),
                indent=2)


def _vSTMD_task(v):
    modelName = 'vSTMD'

    inputpath = _get_inputpath(v)

    '''inference'''
    modelOpt, modelDire, _ = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=TOTAL_FRAME)
    
    # save
    save_json(  {'modelOpt': modelOpt, 'modelDire': modelDire},
                _get_inference_opt_path(modelName, v),
                indent=2)


def _vSTMD_F_task(v):
    modelName = 'vSTMD_F'

    inputpath = _get_inputpath(v)

    '''inference'''
    modelOpt, modelDire, _ = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=TOTAL_FRAME)
    
    # save
    save_json(  {'modelOpt': modelOpt, 'modelDire': modelDire},
                _get_inference_opt_path(modelName, v),
                indent=2)
    

def multi_process_inference(max_workers = 6):
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for v in V_LIST:
            # for tau in TAU_LIST:
            #     futures.append(executor.submit(_ESTMD_task, v, tau))
            #     futures.append(executor.submit(_DSTMD_task, v, tau))
            #     futures.append(executor.submit(_FracSTMD_task, v, tau))
            futures.append(executor.submit(_vSTMD_task, v))
            futures.append(executor.submit(_vSTMD_F_task, v))


        for future in tqdm(concurrent.futures.as_completed(futures), 
                           desc='inference task',
                           total=len(futures)
                           ):
            future.result()


def _load_location_groundtruth(v, timeEnd):
    with open(os.path.join(GROUNDTRUTH_FOLDER, BGR_TYPE,
                        f'ET-Target_Num=1_W=5_H=5_V={v}_L=0-Traj=Ellipse_FPS={FPS}.json'),
        'r') as file:
        data = json.load(file)
    # groundtruth
    posiGT = []
    for tt in range(0, timeEnd):
        GT = data[tt]['bbox']
        posiGT.append([GT,])

    return posiGT

    
def custom_evaluation(modelName, v, tau=None):
    # load modelOpt
    with open(_get_inference_opt_path(modelName, v, tau), 'r') as file:
        data = json.load(file)
    modelOpt = data['modelOpt']
    save_js_file_name = _get_save_result_path(v)

    # load groundtruth
    groundTruth = _load_location_groundtruth(v, TOTAL_FRAME)

    # evaluate
    AUC, AR, AP = evaluate_task(modelOpt, groundTruth, gTError=2, startFrame=100, endFrame=TOTAL_FRAME, plotFigures=False)

    # save
    if tau is not None:
        update_json(f'{modelName}_tau={tau}', {'AUC': AUC, 'AR': AR, 'AP': AP}, save_js_file_name, indent=2)
    else:
        update_json(modelName, {'AUC': AUC, 'AR': AR, 'AP': AP}, save_js_file_name, indent=2)


def multi_process_evaluation(max_workers = 6):
    # with tau
    for model_name in modelNameList[:3]:
        for tau in TAU_LIST:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for v in V_LIST:
                    futures.append(executor.submit(custom_evaluation, model_name, v, tau))

                for future in tqdm(concurrent.futures.as_completed(futures),
                                   desc=f'evaluate {model_name} tau={tau}',
                                   total=len(futures)):
                    future.result()

    # without tau
    for model_name in modelNameList[3:]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for v in V_LIST:
                futures.append(executor.submit(custom_evaluation, model_name, v, None))

            for future in tqdm(concurrent.futures.as_completed(futures), 
                            desc=f'evaluate {model_name}',
                            total=len(futures)
                            ):
                future.result()


def collect_results():
    
    auc_curve_ESTMD = [[None for i in range(len(V_LIST))] for _ in range(len(TAU_LIST))]
    auc_curve_DSTMD = [[None for i in range(len(V_LIST))]  for _ in range(len(TAU_LIST))]
    auc_curve_FracSTMD = [[None for i in range(len(V_LIST))]  for _ in range(len(TAU_LIST))]
    auc_curve_vSTMD = [None for i in range(len(V_LIST))]
    auc_curve_vSTMD_F = [None for i in range(len(V_LIST))]


    for j, v in enumerate(V_LIST):
        with open(_get_save_result_path(v), 'r') as file:
            data = json.load(file)

        for i, tau in enumerate(TAU_LIST):
            auc_curve_ESTMD[i][j] = data[f'ESTMD_tau={tau}']['AUC']
            auc_curve_DSTMD[i][j] = data[f'DSTMD_tau={tau}']['AUC']
            auc_curve_FracSTMD[i][j] = data[f'FracSTMD_tau={tau}']['AUC']

        auc_curve_vSTMD[j] = data['vSTMD']['AUC']
        auc_curve_vSTMD_F[j] = data['vSTMD_F']['AUC']

    
    # 将列表打包到一个字典中
    data = {'auc_curve_ESTMD': auc_curve_ESTMD,  
            'auc_curve_DSTMD': auc_curve_DSTMD, 
            'auc_curve_FracSTMD': auc_curve_FracSTMD, 
            'auc_curve_vSTMD': auc_curve_vSTMD, 
            'auc_curve_vSTMD_F': auc_curve_vSTMD_F,
            'v_list': V_LIST,
            'tau_list': TAU_LIST,
            }

    # 保存到 JSON 文件
    save_json(data, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'velocity_AUC_curve.json'))

    print('\n\nDone...')
    


if __name__ == '__main__':    
    # multi_process_inference(12)
    multi_process_evaluation(12)
    collect_results()


