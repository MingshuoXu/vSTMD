import os
import sys
ITEM_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ITEM_FOLDER)
import concurrent.futures
from math import atan2
import time


import numpy as np
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import pandas as pd
from skopt import gp_minimize
from skopt.utils import dump, load
import torch.nn.functional as F


# Add the path to the package containing the models
import config
from config import ristDatasetPath
from smalltargetmotiondetectors.api import evaluate_task # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader # type: ignore
from smalltargetmotiondetectors.model.vstmd import vSTMD # type: ignore



LOG_PATH = os.path.join(ITEM_FOLDER, "parameter_analysis", "bo_log.txt")

def tensor_to_sparse_list(tensor):
    
    # Get the indices and values of non-zero elements
    nz = tensor.nonzero(as_tuple=False)
    if nz.numel() == 0:
        return []
    
    y_idx = nz[:, -2].tolist()
    x_idx = nz[:, -1].tolist()
    values = tensor[0, 0, y_idx, x_idx]
    v_list = values.tolist()

    # Pack into list of lists
    sparse_list = [
        [x, y, v]
        for y, x, v in zip(y_idx, x_idx, v_list)
    ]

    return sparse_list


def torch_nms(input_torch, window_size=8):
    
    # 使用最大池化获取局部最大值
    local_max = F.max_pool2d(
        input_torch, 
        kernel_size=window_size*2+1, 
        stride=1, 
        padding=window_size
    )
    

    return input_torch * (input_torch == local_max)


def instance_model(gLeak, alpha):
    model = vSTMD(device='cuda')
    model.set_parameter(gLeak=gLeak, alpha=alpha)
    model.init_config()

    return model


def custom_vSTMD_forward(model, input_img):
    input_img = torch.from_numpy(input_img).to(device='cuda').float().unsqueeze(0).unsqueeze(0)
        
    # Perform inference using the model
    result, _ = model.process(input_img)
    torch.cuda.synchronize()

    # response
    response_tensor = result['response']
    if torch.max(response_tensor) == 0:
        return [], []

    response_tensor = torch_nms(response_tensor)
    response_tensor /= torch.max(response_tensor)
    response_array = tensor_to_sparse_list(response_tensor)

    # direction
    direction_torch = result['direction']
    if direction_torch is not None:
        direction_array = [[y, x, float(direction_torch[0,0,x,y])] for y, x, _ in response_array]
    else:
        direction_array = []

    return response_array, direction_array


def inference_task(model, input_stream):

    # inference
    responses = []
    directions = []
    while input_stream.hasFrame:
        # Read the next frame from the video stream
        grayImg, _ = input_stream.get_next_frame()

        response_array, direction_array = custom_vSTMD_forward(model, grayImg)
        responses.append(response_array)
        directions.append(direction_array)

    return responses, directions


def evaluate_direction_task(respResults, direResluts, bboxData, directions, startFrame, endFrame):

    def calc_direction_error(respRes, direRes, bbox, gtDire):
        """
        Calculate the absolute angular error between the response results and the ground truth direction.
        """
        x, y, w, h = bbox[0]
        
        filtered_pairs = [(a_row[2], b_row[2]) for a_row, b_row in zip(respRes, direRes) 
                          if (x - 1 <= a_row[0] <= x + w + 1) and (y - 1 <= a_row[1] <= y + h + 1)
        ]

        if len(filtered_pairs):  # 如果有满足条件的元素
            _, dire = max(filtered_pairs, key=lambda x: x[0])
            AE = abs(dire - gtDire)
            # Ensure AAE is in the range [0, pi]
            resAE = AE if AE < np.pi else 2 * np.pi - AE
            return resAE
        else:
            return None
            
    accAE = []
    for i in range(startFrame, endFrame):
        if len(respResults[i]) == 0:
            diError = None
        else:
            diError = calc_direction_error(
                respResults[i],
                direResluts[i],
                bboxData[i],
                directions[i-1]
            )
        if diError is not None:
            accAE.append(diError)  
    AAE = np.mean(np.array(accAE))  # Average Angular Error
    AAE = AAE if AAE < np.pi else 2 * np.pi - AAE  # Ensure AAE is in the range [0, pi]
    return AAE


def prepare_groundtruth(dataset_name):
    with open(os.path.join(ristDatasetPath, dataset_name, f'{dataset_name}_annotation.json'),'r') as file:
        groundTruth = json.load(file)
    
    bboxData = []
    directions = []
    for frame_data in groundTruth['frames']:
        
        bbox = frame_data['objects']['bbox']
        bboxData.append([bbox,])  # bbox is in [x, y, w, h] 
        
        motion_vector = frame_data['objects']['motion_vector']
        if len(motion_vector) == 0:
            direction = None
        else:
            direction = np.mod(atan2(- motion_vector[1], motion_vector[0]), 2 * np.pi)
        directions.append(direction)  # direction in radians
    return bboxData, directions


def vSTMD_task(gLeak, alpha, dataset_name):

    ## instantiate model
    model = instance_model(gLeak, alpha)
    input_stream = VidstreamReader(os.path.join(ristDatasetPath, dataset_name, f'{dataset_name}.mp4'))

    ## inference
    model_responses, model_directions = inference_task(model, input_stream)
    torch.cuda.synchronize()
    del model, input_stream

    ## evaluation
    # groundtruth
    bboxs, gt_directions = prepare_groundtruth(dataset_name)

    aucOfROC, AR, AP = evaluate_task(model_responses, bboxs, startFrame=1, endFrame=len(model_responses), plotFigures=False)
    AAE = evaluate_direction_task(model_responses, model_directions, bboxs, gt_directions, 1, len(model_directions))

    return aucOfROC, AR, AP, float(AAE)


# dataset information
datasetInfo = {
    'GX010071-1': list(range(1300)),
    'GX010220-1': list(range(1300)),
    'GX010228-1': list(range(1300)),
    'GX010230-1': list(range(2400)),
    'GX010231-1': list(range(2400)),
    'GX010241-1': list(range(3600)),
    'GX010250-1': list(range(2000)),
    'GX010266-1': list(range(2400)),
    'GX010290-1': list(range(1300)),
    'GX010291-1': list(range(1300)),
    'GX010303-1': list(range(2400)),
    'GX010307-1': list(range(1000)),
    'GX010315-1': list(range(1000)),
    'GX010321-1': list(range(1000)),
    'GX010322-1': list(range(1300)),
    'GX010327-1': list(range(900)),
    'GX010335-1': list(range(1300)),
    'GX010336-1': list(range(1000)),
    'GX010337-1': list(range(700)),
}


def para_to_performance(gLeak, alpha):
    roc, ar, ap, aae = 0, 0, 0, 0


    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        
        for datasetName in datasetInfo.keys():      
            futures.append(executor.submit(vSTMD_task, gLeak, alpha, datasetName))
        

        for future in concurrent.futures.as_completed(futures):
            aucOfROC, AR, AP, AAE = future.result()
            roc += aucOfROC
            ar += AR
            ap += AP
            aae += AAE

    
    return roc/len(datasetInfo), ar/len(datasetInfo), ap/len(datasetInfo), aae/len(datasetInfo)


def optional_func(para):

    gLeak, alpha = para
    aucOfROC, AR, AP, AAE = para_to_performance(gLeak, alpha)

    score = (1.5 - aucOfROC) * (AAE + 0.5)  # Example: maximize ROC, AR, AP and minimize AAE
    
    if np.isnan(score):
        return 1000
    else:
        with open(LOG_PATH, 'a') as f:
            f.write(f"\t-----------\n\tauc={aucOfROC:.4f}, AR={AR:.4f}, AP={AP:.4f}, AAE={AAE:.4f}\n")
        return score  # since we want to minimize


def grid_search(a_range, b_range):
    results = []

    print(f"Searching {len(a_range)} x {len(b_range)} = {len(a_range)*len(b_range)} configs ...")

    for g_Leak in tqdm(a_range):
        for alpha in b_range:
            res = para_to_performance(g_Leak, alpha)
            score = res[0] + res[1] + res[2] - res[3]  # Example: maximize ROC, AR, AP and minimize AAE

            results.append([float(g_Leak), float(alpha), float(score)])

    # 保存所有参数组合
    df = pd.DataFrame(results, columns=["g_Leak", "alpha", "score"])

    # 自动找到最优
    idx = df["score"].idxmax()
    best = df.iloc[idx]

    print("\n===== Best Parameter Found =====")
    print(best)

    return df, best


def course_to_fine_search():
    # first coarse search
    ALPHA_LIST = [i*0.1 for i in range(1, 11)]
    G_LEAK_LIST = [i*0.1 for i in range(11)]
    df1, best1 = grid_search(G_LEAK_LIST, ALPHA_LIST)

    # second fine search
    a0, b0 = best1["a"], best1["b"]
    print("\nRefining around region:", a0, b0)

    fine_a = np.linspace(a0 - 0.1, a0 + 0.1, 41)
    fine_b = np.linspace(b0 - 0.1, b0 + 0.1, 41)
    fine_a = np.clip(fine_a, 0, 1)
    fine_b = np.clip(fine_b, 0, 1)

    df2, best2 = grid_search(fine_a, fine_b)

    return df1, best1, df2, best2


def unit_test():
    gLeak = 0.5
    alpha = 0.3

    aucOfROC, AR, AP, AAE = para_to_performance(gLeak, alpha)
    print('G_leak: %.2f' %gLeak, 'Alpha: %.2f' %alpha
          , 'AUC of ROC: %.4f' %aucOfROC
          , 'AR: %.4f' %AR
          , 'AP: %.4f' %AP
          , 'AAE: %.4f' %AAE
          )


class OptimizationCallback:
    def __init__(self, log_file_path, item_folder):
        self.last_time = time.time()
        self.log_file_path = log_file_path
        self.item_folder = item_folder
        
    def __call__(self, res):
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        
        i = len(res.x_iters) - 1
        
        data = f"iter-{i}: g_Leak={float(res.x_iters[i][0]):.3f}, alpha={float(res.x_iters[i][1]):.3f}, score={float(res.func_vals[i]):.3f}, time={elapsed:.1f}s"
        
        with open(self.log_file_path, 'a') as f:
            f.write(data+'\n')
        
        print_data = f"{data}. \t Best so far: g_Leak={float(res.x[0]):.3f}, alpha={float(res.x[1]):.3f}, score={float(res.fun):.3f}"
        print(print_data)
        
        dump(res, os.path.join(self.item_folder, "parameter_analysis", "bo_state.pkl"))


def bayes_option():
    with open(LOG_PATH, 'w') as f:
        f.write('')

    on_step_callback = OptimizationCallback(LOG_PATH, ITEM_FOLDER)
    print("Starting Bayesian Optimization...")
    res = gp_minimize(
        func = optional_func,              # the function to minimize
        dimensions=[(0.01, 0.99), (0.01, 0.99)],  # the bounds on each dimension of x
        acq_func="EI",             # the acquisition function
        n_calls=500,                 # the number of evaluations of f
        n_random_starts=100,         # the number of random initialization points
        random_state=25,            # the random seed
        callback=on_step_callback,         # callback function to log progress
        )

    print("\n===== FINAL BEST RESULT =====")
    print("Best params:", res.x)
    print("Best score:", -res.fun)

    with open(LOG_PATH, 'a') as f:
        f.write("\n===== FINAL BEST RESULT =====\n")
        f.write(f"Best params: g_Leak={float(res.x[0]):.3f}, alpha={float(res.x[1]):.3f}\n")
        f.write(f"Best score: {float(-res.fun):.3f}\n")


if __name__ == '__main__':
    # df1, best1, df2, best2 = main_search()
    bayes_option()



