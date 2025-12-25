import os
import sys
ITEM_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ITEM_FOLDER)
import concurrent.futures
from math import atan2
import time
import platform


import numpy as np
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import pandas as pd
from skopt import gp_minimize
from skopt.utils import dump, load
import torch.nn.functional as F
import seaborn as sns
import re
from scipy.interpolate import Rbf

# Add the path to the package containing the models
import config
if platform.system() == 'Linux':
    ristDatasetPath = os.path.join('/mnt', 'windows_D', 'STMD_Dataset', 'RIST')
else:
    # dataset path
    ristDatasetPath = os.path.join('D:/', 'STMD_Dataset', 'RIST')
from smalltargetmotiondetectors.api import evaluate_task # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader # type: ignore
from smalltargetmotiondetectors.model.vstmd import vSTMD, vSTMD_F # type: ignore



LOG_PATH = os.path.join(FILE_FOLDER, "bo_log.txt")
CACHE_FILE = os.path.join(FILE_FOLDER, 'processed_data.csv')


def _tensor_to_sparse_list(tensor):
    
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


def _torch_nms(input_torch, window_size=8):
    
    # 使用最大池化获取局部最大值
    local_max = F.max_pool2d(
        input_torch, 
        kernel_size=window_size*2+1, 
        stride=1, 
        padding=window_size
    )
    

    return input_torch * (input_torch == local_max)


def _instance_model(gLeak, alpha):
    model_vSTMD = vSTMD(device='cuda')
    model_vSTMD.set_parameter(gLeak=gLeak, alpha=alpha)
    model_vSTMD.init_config()

    model_vSTMD_F = vSTMD_F(device='cuda')
    model_vSTMD_F.set_parameter(gLeak=gLeak, alpha=alpha)
    model_vSTMD_F.init_config()

    return model_vSTMD, model_vSTMD_F


def _custom_vSTMD_forward(model, input_img):
        
    # Perform inference using the model
    result, _ = model.process(input_img)

    # response
    response_tensor = result['response']
    if torch.max(response_tensor) == 0:
        return [], []

    response_tensor = _torch_nms(response_tensor)
    response_tensor /= torch.max(response_tensor)
    response_array = _tensor_to_sparse_list(response_tensor)

    # direction
    direction_torch = result['direction']
    if direction_torch is not None:
        direction_array = [[y, x, float(direction_torch[0,0,x,y])] for y, x, _ in response_array]
    else:
        direction_array = []

    return response_array, direction_array


def _inference_task(input_stream, model_vSTMD, model_vSTMD_F):

    # inference
    responses_vSTMD = []
    directions_vSTMD = []
    responses_vSTMD_F = []
    directions_vSTMD_F = []
    while input_stream.hasFrame:
        # Read the next frame from the video stream
        grayImg, _ = input_stream.get_next_frame()
        input_img = torch.from_numpy(grayImg).to(device='cuda').float().unsqueeze(0).unsqueeze(0)
        
        response_array1, direction_array1 = _custom_vSTMD_forward(model_vSTMD, input_img)
        response_array2, direction_array2 = _custom_vSTMD_forward(model_vSTMD_F, input_img)

        responses_vSTMD.append(response_array1)
        directions_vSTMD.append(direction_array1)
        responses_vSTMD_F.append(response_array2)
        directions_vSTMD_F.append(direction_array2)

    return responses_vSTMD, directions_vSTMD, responses_vSTMD_F, directions_vSTMD_F


def _evaluate_direction_task(respResults, direResluts, bboxData, directions, startFrame, endFrame):

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
    if len(accAE) == 0:
        return np.pi  # If no valid angular errors, return maximum error
     
    AAE = np.nanmean(np.array(accAE))  # Average Angular Error
    AAE = AAE if AAE < np.pi else 2 * np.pi - AAE  # Ensure AAE is in the range [0, pi]
    return AAE


def _prepare_groundtruth(dataset_name):
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


def _vSTMD_task(gLeak, alpha, dataset_name):

    ## instantiate model
    model_vSTMD, model_vSTMD_F = _instance_model(gLeak, alpha)
    input_stream = VidstreamReader(os.path.join(ristDatasetPath, dataset_name, f'{dataset_name}.mp4'))

    ## inference
    vSTMD_responses, vSTMD_directions, vSTMD_F_responses, vSTMD_F_directions = \
          _inference_task(input_stream, model_vSTMD, model_vSTMD_F)
    del model_vSTMD, model_vSTMD_F, input_stream

    ## evaluation
    # groundtruth
    bboxs, gt_directions = _prepare_groundtruth(dataset_name)

    vSTMD_AUC, vSTMD_AR, vSTMD_AP = evaluate_task(vSTMD_responses, bboxs, startFrame=1, endFrame=len(vSTMD_responses), plotFigures=False)
    vSTMD_AAE = _evaluate_direction_task(vSTMD_responses, vSTMD_directions, bboxs, gt_directions, 1, len(vSTMD_directions))

    vSTMD_F_AUC, vSTMD_F_AR, vSTMD_F_AP = evaluate_task(vSTMD_F_responses, bboxs, startFrame=1, endFrame=len(vSTMD_F_responses), plotFigures=False)
    vSTMD_F_AAE = _evaluate_direction_task(vSTMD_F_responses, vSTMD_F_directions, bboxs, gt_directions, 1, len(vSTMD_F_directions))

    return vSTMD_AUC, vSTMD_AR, vSTMD_AP, float(vSTMD_AAE), vSTMD_F_AUC, vSTMD_F_AR, vSTMD_F_AP, float(vSTMD_F_AAE)


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
    roc_vSTMD, ar_vSTMD, ap_vSTMD, aae_vSTMD = 0, 0, 0, 0
    roc_vSTMD_F, ar_vSTMD_F, ap_vSTMD_F, aae_vSTMD_F = 0, 0, 0, 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for datasetName in datasetInfo.keys():      
            futures.append(executor.submit(_vSTMD_task, gLeak, alpha, datasetName))
        
        for future in concurrent.futures.as_completed(futures):
            vSTMD_AUC, vSTMD_AR, vSTMD_AP, vSTMD_AAE, vSTMD_F_AUC, vSTMD_F_AR, vSTMD_F_AP, vSTMD_F_AAE = future.result()
            roc_vSTMD += vSTMD_AUC
            ar_vSTMD += vSTMD_AR
            ap_vSTMD += vSTMD_AP
            aae_vSTMD += vSTMD_AAE
            roc_vSTMD_F += vSTMD_F_AUC
            ar_vSTMD_F += vSTMD_F_AR
            ap_vSTMD_F += vSTMD_F_AP
            aae_vSTMD_F += vSTMD_F_AAE
    
    return (roc_vSTMD / len(datasetInfo), 
            ar_vSTMD / len(datasetInfo),
            ap_vSTMD / len(datasetInfo),
            aae_vSTMD / len(datasetInfo),
            roc_vSTMD_F / len(datasetInfo),
            ar_vSTMD_F / len(datasetInfo),
            ap_vSTMD_F / len(datasetInfo),
            aae_vSTMD_F / len(datasetInfo)
           )


def optional_func(para):

    gLeak, alpha = para
    vSTMD_AUC, vSTMD_AR, vSTMD_AP, vSTMD_AAE, vSTMD_F_AUC, vSTMD_F_AR, vSTMD_F_AP, vSTMD_F_AAE = para_to_performance(gLeak, alpha)

    save_data = {
        'g_Leak': float(gLeak),
        'alpha': float(alpha),
        'vSTMD_AUC': float(vSTMD_AUC),
        'vSTMD_AR': float(vSTMD_AR),
        'vSTMD_AP': float(vSTMD_AP),
        'vSTMD_AAE': float(vSTMD_AAE),
        'vSTMD_F_AUC': float(vSTMD_F_AUC),
        'vSTMD_F_AR': float(vSTMD_F_AR),
        'vSTMD_F_AP': float(vSTMD_F_AP),
        'vSTMD_F_AAE': float(vSTMD_F_AAE),
    }
    # --- 实时存入 JSONL ---
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        # 将字典转为字符串并换行
        f.write(json.dumps(save_data) + '\n')
        

    if vSTMD_AAE > 1 or vSTMD_AUC < 0.3:
        score = 10
    else:
        score = (1-vSTMD_AUC) * (1-vSTMD_AR) * (1-vSTMD_AP) * vSTMD_AAE \
              * (1-vSTMD_F_AUC) * (1-vSTMD_F_AR) * (1-vSTMD_F_AP) * vSTMD_F_AAE  # Example: maximize ROC, AR, AP and minimize AAE
    
    if np.isnan(score):
        return 100
    else:
        return score  # since we want to minimize


class OptimizationCallback:
    def __init__(self, log_file_path, file_folder):
        self.last_time = time.time()
        self.log_file_path = log_file_path
        self.file_folder = file_folder
        
    def __call__(self, res):
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        
        i = len(res.x_iters) - 1
        
        data = f"iter-{i}: g_Leak={float(res.x_iters[i][0]):.3f}, alpha={float(res.x_iters[i][1]):.3f}, " + \
            f"score={float(res.func_vals[i]):.3f}, time={elapsed:.1f}s." + \
            f"\t Best so far: g_Leak={float(res.x[0]):.3f}, alpha={float(res.x[1]):.3f}, score={float(res.fun):.3f}"
        
        print(data)
        
        dump(res, os.path.join(self.file_folder, "bo_state.pkl"))


def main_bayes_option():
    checkpoint_path = os.path.join(FILE_FOLDER, "bo_state.pkl")
    
    # 1. 尝试加载上一次的状态
    if os.path.exists(checkpoint_path):
        print(f"检测到断点文件: {checkpoint_path}，正在恢复...")
        res_old = load(checkpoint_path)
        x0 = res_old.x_iters
        y0 = res_old.func_vals
        print(f"成功加载 {len(x0)} 条历史记录。")
    else:
        # 如果没有断点，则执行你原来的网格点初始化逻辑
        print("未检测到断点，执行初始网格搜索...")
        x1_range = np.linspace(0.01, 0.99, 3)
        x2_range = np.linspace(0.01, 0.99, 3)
        x0 = [[x1, x2] for x1 in x1_range for x2 in x2_range]
        y0 = None # 第一次运行，没有对应的结果

    on_step_callback = OptimizationCallback(LOG_PATH, FILE_FOLDER)

    # 2. 调用 gp_minimize
    # 注意：n_calls 是总迭代次数（包括已经完成的部分）
    total_calls = 500 
    
    res = gp_minimize(
        func=optional_func,
        dimensions=[(0.01, 0.99), (0.01, 0.99)],
        x0=x0,              
        y0=y0,              
        n_calls=total_calls,                 
        n_random_starts=50, 
        callback=on_step_callback,
    )

    print("\n===== FINAL BEST RESULT =====")
    print("Best params:", res.x)
    print("Best score:", -res.fun)



def get_data(log_file):
    # 1. 检查缓存文件是否存在
    if os.path.exists(CACHE_FILE):
        print(f"正在从缓存加载数据: {CACHE_FILE}...")
        return pd.read_csv(CACHE_FILE)
    
    # 2. 如果不存在，执行解析逻辑
    print(f"未发现缓存，正在解析原始日志 {log_file}，请稍候...")
    extracted_data = []
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"找不到原始日志文件: {log_file}")

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                raw = json.loads(line)
                item = {
                    "alpha": raw.get("alpha"),
                    "g_Leak": raw.get("g_Leak"),
                    "vSTMD_AUC": raw.get("vSTMD_AUC"),
                    "vSTMD_F_AUC": raw.get("vSTMD_F_AUC") or raw.get("vSTMD_AUC_F"),
                    "vSTMD_AAE": raw.get("vSTMD_AAE"),
                    "vSTMD_F_AAE": raw.get("vSTMD_F_AAE") or raw.get("vSTMD_AAE_F")
                }
                extracted_data.append(item)
            except Exception:
                continue
    
    df = pd.DataFrame(extracted_data)
    
    # 3. 保存到本地，下次直接用
    df.to_csv(CACHE_FILE, index=False)
    print(f"解析完成，提取了 {len(df)} 条数据，已存入 {CACHE_FILE}")
    
    return df


def draw_contour_grid(df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    
    # 1. 预先计算每一行的全局值域，确保 Colorbar 准确
    # 第一行 (AUC) 的值域
    auc_min = min(df['vSTMD_AUC'].min(), df['vSTMD_F_AUC'].min())
    auc_max = max(df['vSTMD_AUC'].max(), df['vSTMD_F_AUC'].max())
    
    # 第二行 (AAE) 的值域
    aae_min = min(df['vSTMD_AAE'].min(), df['vSTMD_F_AAE'].min())
    aae_max = min(1.5, max(df['vSTMD_AAE'].max(), df['vSTMD_F_AAE'].max()))

    plot_map = [
        {"pos": (0, 0), "col": "vSTMD_AUC", "title": "vSTMD: AUC", "cmap": "viridis", "range": (auc_min, auc_max)},
        {"pos": (0, 1), "col": "vSTMD_F_AUC", "title": "vSTMD-F: AUC", "cmap": "viridis", "range": (auc_min, auc_max)},
        {"pos": (1, 0), "col": "vSTMD_AAE", "title": "vSTMD: AAE", "cmap": "viridis_r", "range": (aae_min, aae_max)},
        {"pos": (1, 1), "col": "vSTMD_F_AAE", "title": "vSTMD-F: AAE", "cmap": "viridis_r", "range": (aae_min, aae_max)}
    ]

    for config in plot_map:
        ax = axes[config["pos"]]
        vmin, vmax = config["range"]
        
        # 插值逻辑... (保持不变)
        x, y, z = df['alpha'].values, df['g_Leak'].values, df[config["col"]].values
        rbf = Rbf(x, y, z, function='multiquadric', smooth=0.1)
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        XI, YI = np.meshgrid(xi, yi)
        ZI = rbf(XI, YI)

        # 【关键改进】：传入 vmin 和 vmax，并手动指定 levels
        # 使用 np.linspace 确保两张图的等高线层级完全一致
        levels = np.linspace(vmin, vmax, 50)
        cp = ax.contourf(XI, YI, ZI, levels=levels, cmap=config["cmap"], extend='both')
        
        # 绘制白色等高线 (也可以统一间隔)
        if config["pos"][0] == 0:
            line_levels = np.linspace(vmin, vmax, int((vmax-vmin)/0.05))
        else:
            line_levels = np.linspace(vmin, vmax, int((vmax-vmin)/0.1))
        line_colors = ax.contour(XI, YI, ZI, levels=line_levels, colors='#333333', linewidths=0.6, alpha=0.5)
        ax.clabel(line_colors, inline=True, fontsize=8, fmt='%.2f')

        # 辅助装饰...
        ax.set_title(config["title"], fontsize=12, pad=10)
        if config["pos"][0] == 1: ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$g_{Leak}$')
        
        # 2. 每行只在第二列添加 Colorbar
        if config["pos"][1] == 1:
            # 这里的 cp 包含了该行的全局 vmin/vmax 信息
            cbar = fig.colorbar(cp, ax=axes[config["pos"][0], :], shrink=0.8, aspect=30)
            cbar.ax.tick_params(labelsize=9)

    plt.suptitle('Parameter Sensitivity Analysis: AUC vs AAE', fontsize=14)
    plt.savefig(os.path.join(FILE_FOLDER, 'parameter_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    # main_bayes_option()

    df_data = get_data(os.path.join(FILE_FOLDER, "bo_log_total.txt"))
    draw_contour_grid(df_data)
    




