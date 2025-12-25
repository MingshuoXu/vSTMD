import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.transforms as transforms

# --- 配置路径 ---
rist_dataset_path = r'D:\STMD_Dataset\RIST'
curr_pth = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(curr_pth, 'visulize_data.json')

# 数据集信息
dataset_info = [
    ['GX010230-1', range(0, 2400)],
    ['GX010241-1', range(0, 3600)],
    ['GX010266-1', range(0, 2400)],
    ['GX010290-1', range(0, 1300)],
    ['GX010303-1', range(0, 2400)],
    ['GX010307-1', range(0, 1000)],
    ['GX010337-1', range(0, 700)],
]

opticflow_model_list = ['RAFT', 'MemFlow', 'StreamFlow', 'DpFlow', 'FlowDiffuser']
directional_stmd_list = ['STMDPlus', 'ApgSTMD', 'vSTMD', 'vSTMD_F']
model_list = opticflow_model_list + directional_stmd_list

# 加载 JSON 数据
with open(json_file_path, 'r') as f:
    file_data = json.load(f)

# --- 绘图辅助函数 ---

def read_last_img(path, frame_num):
    cap = cv2.VideoCapture(path)
    # OpenCV 帧索引从0开始
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.zeros((100, 100, 3), dtype=np.uint8)

def custom_plot(ax, x, y, z):
    """轨迹线绘制 (HSV 颜色)"""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 1. 获取 hsv 颜色表的一个副本
    cmap = plt.get_cmap('hsv').copy()
    
    # 2. 设置 NaN 值的颜色。
    cmap.set_bad(color='gray', alpha=0.03) 

    norm = plt.Normalize(0, 2 * np.pi)
    
    # 3. 使用修改后的 cmap
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    
    # 如果 z 的长度和 x, y 一致，通常需要取 z[:-1] 或计算线段中点值
    # 这里假设 z 的长度已经和线段数 (len(x)-1) 匹配
    lc.set_array(z)
    lc.set_linewidth(2)
    
    ax.add_collection(lc)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.autoscale()


def show_criteria(ax, dataset_name, model_name):
    """显示评估指标文本"""
    # 模拟路径逻辑
    base_dir = os.path.dirname(os.path.dirname(curr_pth))
    eval_path = os.path.join(base_dir, 'evaluate_result', 'RIST', f'{dataset_name}.json')
    blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    try:
        with open(eval_path, 'r') as f:
            data = json.load(f)
        aae = data[model_name].get('AAE', 0)
        auc = f"{data[model_name]['AUC']*100:.1f}%" if 'AUC' in data[model_name] else None
        if auc is None:
            ax.text(0.5, -0.05, f"AAE:{aae:.2f}", 
                transform=ax.transAxes, ha='center', fontsize=8)
        else:
            ax.text(0.5, -0.05, f"AAE:{aae:.2f}, AUC:{auc}", 
                transform=ax.transAxes, ha='center', fontsize=8)
    except:
        pass

# --- 主绘图逻辑 ---
def main_python():
    num_datasets = len(dataset_info)
    num_models = len(model_list)
    # 行数 = 模型数 + 2 (Raw & GT), 列数 = 数据集数 + 1 (左侧标题列)
    n_rows = num_models + 2
    n_cols = num_datasets + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(num_datasets*1.5, n_rows*0.8), constrained_layout=True)

    # 1. 第一行：Raw Image
    axes[0, 0].text(0.5, 0.5, 'raw image', ha='center', va='center')
    axes[0, 0].axis('off')
    for i, (d_name, f_range) in enumerate(dataset_info):
        ax = axes[0, i+1]
        v_path = os.path.join(rist_dataset_path, d_name, f"{d_name}.mp4")
        img = read_last_img(v_path, f_range[-1])
        ax.imshow(img)
        ax.set_title(d_name[:-2], fontsize=9)
        ax.axis('off')

    # 2. 第二行：Ground Truth
    axes[1, 0].text(0.5, 0.5, 'groundtruth', ha='center', va='center')
    axes[1, 0].axis('off')
    for i, (d_name, f_range) in enumerate(dataset_info):
        ax = axes[1, i+1]
        gt_key = f"{d_name[:-2]}-1_groundtruth"
        if gt_key in file_data.keys():
            gt = file_data[gt_key]
            locs = np.array(gt['location'])
            dirs = np.array(gt['direction'])
            custom_plot(ax, locs[f_range[0]:f_range[-1], 0], locs[f_range[0]:f_range[-1], 1], dirs[f_range[0]:f_range[-1]])

    # 3. 后续行：各个模型的结果
    for j, m_name in enumerate(model_list):
        row_idx = j + 2
        axes[row_idx, 0].text(0.5, 0.5, m_name.replace('_', '-'), ha='center', va='center')
        axes[row_idx, 0].axis('off')
        
        for i, (d_name, f_range) in enumerate(dataset_info):
            ax = axes[row_idx, i+1]
            res_key = f"{d_name[:-2]}-1_{m_name}"
            if res_key in file_data:
                res = file_data[res_key]
                resp = np.array(res['response'])
                dirs = np.array(res['directions'])
                
                # 计算起始帧逻辑
                large_plot = len(dirs)
                frame00 = max(0, len(resp) - large_plot)
                
                custom_plot(ax, resp[frame00:, 0], resp[frame00:, 1], dirs)
                show_criteria(ax, d_name, m_name)
            else:
                ax.axis('off')

    plt.show()

def main():
    num_datasets = len(dataset_info)
    num_models = len(model_list)
    # 行数 = 模型数 + 2 (Raw & GT), 列数 = 数据集数 + 1 (左侧标题列)
    n_rows = num_models + 1
    n_cols = num_datasets

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(num_datasets*1.5, n_rows*0.8), constrained_layout=True)

    # 2. 第二行：Ground Truth
    # axes[1, 0].text(0.5, 0.5, 'groundtruth', ha='center', va='center')
    for i, (d_name, f_range) in enumerate(dataset_info):
        ax = axes[0, i]
        gt_key = f"{d_name[:-2]}-1_groundtruth"
        if gt_key in file_data.keys():
            gt = file_data[gt_key]
            locs = np.array(gt['location'])
            dirs = np.array(gt['direction'])
            custom_plot(ax, locs[f_range[0]:f_range[-1], 0], locs[f_range[0]:f_range[-1], 1], dirs[f_range[0]:f_range[-1]])

    # 3. 后续行：各个模型的结果
    for j, m_name in enumerate(model_list):
        row_idx = j + 1
        # axes[row_idx, 0].text(0.5, 0.5, m_name.replace('_', '-'), ha='center', va='center')
        # axes[row_idx, 0].axis('off')
        
        for i, (d_name, f_range) in enumerate(dataset_info):
            ax = axes[row_idx, i]
            res_key = f"{d_name[:-2]}-1_{m_name}"
            if res_key in file_data:
                res = file_data[res_key]
                resp = np.array(res['response'])
                dirs = np.array(res['directions'])
                
                # 计算起始帧逻辑
                large_plot = len(dirs)
                frame00 = max(0, len(resp) - large_plot)
                
                custom_plot(ax, resp[frame00:, 0], resp[frame00:, 1], dirs)
            else:
                ax.axis('off')

    
    plt.savefig(os.path.join(curr_pth, 'rist_all_models.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    # main()

    main_python()