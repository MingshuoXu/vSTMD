import os
import sys
ITEM_PTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ITEM_PTH)
from copy import deepcopy
from collections import deque

import json
from tqdm import tqdm
from matplotlib import pyplot as plt
# 全局设置字体
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体家族
plt.rcParams['font.size'] = 13                   # 设置字体大小
plt.rcParams['axes.titlesize'] = 13              # 标题字体大小
plt.rcParams['axes.labelsize'] = 13              # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12             # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12             # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12             # 图例字体大小
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
import numpy as np

import config
from smalltargetmotiondetectors.model import backbone, vstmd # type: ignore
from smalltargetmotiondetectors.util.iostream import ImgstreamReader # type: ignore
from utils import custom_serialize



JSON_FILE_PATH = os.path.realpath(__file__).replace('.py', '.json')



def get_input_stream(velocity):
    ''' Dynamically create a video stream reader or other input type '''
    input_template = os.path.join('D:/', 'STMD_Dataset', 'vSTMD_Panorama_Stimuli', 'White-Background',
                             f'TW-0.8d-TH-0.8d-TV-{velocity}_pixel_s-TL-0-SamFre-1000',
                             'WhiteBG*.tif')
    objIptStream = ImgstreamReader(input_template)

    return objIptStream


class Delay_And_Correlate(backbone.ESTMD):
    def __init__(self):
        super().__init__(device='cpu')
        self.off_signal_list = deque(maxlen=13)

    def process(self, iptMatrix):
        # Process input matrix through model components
        retinaOpt = self.hRetina.process(iptMatrix)
        laminaOpt = self.hLamina.process(retinaOpt)

        self.on_signal = np.maximum(laminaOpt, 0)
        self.off_signal = np.maximum(-laminaOpt, 0)

        if len(self.off_signal_list) == self.off_signal_list.maxlen:
            self.delay_off = self.off_signal_list.popleft()
        else:
            self.delay_off = np.zeros_like(self.off_signal)

        self.off_signal_list.append(deepcopy(self.off_signal))

        self.out = self.on_signal * self.delay_off

    
class cIDP_location(vstmd.vSTMD):
    def __init__(self):
        super().__init__(device='cpu')

    def process(self, iptMatrix):
        # Process input matrix through model components
        remina_opt = self.hRetina.process(iptMatrix)
        self.L_on, self.L_off = self.hLamina.process(remina_opt)
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

    with open(JSON_FILE_PATH, 'w') as f:
        data = {'delay_and_correlate_results': delay_and_correlate_results,
                'cIDP_loc_results': cIDP_loc_results}
        data = custom_serialize(data)
        f.write(data)


def show_results1():
    with open(JSON_FILE_PATH, 'r') as f:
        data = json.load(f)
    
    delay_and_correlate_results = data['delay_and_correlate_results']
    cIDP_loc_results = data['cIDP_loc_results']

    fig, axs = plt.subplots(4, 2, figsize=(5, 7))
    v_list = [500, 1000, 2000, 3000]
    plot_range = [385, 193, 96, 64]
    x_ticks = np.arange(40)
 

    # 统一的线条样式
    L_ON_STYLE = {'linestyle': '-', 'color': 'red', 'linewidth': 1.5}
    L_OFF_STYLE = {'linestyle': '-', 'color': 'blue', 'linewidth': 1.5}
    OUTPUT_STYLE = {'linestyle': '-', 'color': 'black', 'linewidth': 2.5}
    
    # 存储图例句柄和标签
    all_handles = []
    all_labels = []
    added_labels = set()  # 避免重复标签

    for i, v in enumerate(v_list):
        ax_left = axs[i, 0]
        out_line = np.array(delay_and_correlate_results[f'{v}']['out'][plot_range[i]:plot_range[i]+40])
        
        l2 = ax_left.plot(x_ticks, delay_and_correlate_results[f'{v}']['delay_off'][plot_range[i]:plot_range[i]+40], 
                         linestyle=':', color='blue', linewidth=1.5)[0]
        l3 = ax_left.plot(x_ticks, delay_and_correlate_results[f'{v}']['L_on'][plot_range[i]:plot_range[i]+40], 
                         **L_ON_STYLE)[0]
        l4 = ax_left.plot(x_ticks, delay_and_correlate_results[f'{v}']['L_off'][plot_range[i]:plot_range[i]+40], 
                         **L_OFF_STYLE)[0]
        l1 = ax_left.plot(x_ticks, out_line*10, 
                         **OUTPUT_STYLE)[0]
        
        if i == 0:
            ax_left.set_title('Delay and Correlate')
        if i == 3:
            ax_left.set_xlabel('Time (ms)')
        ax_left.set_ylabel('Response')
        ax_left.grid(True, alpha=0.3, linestyle='--')
        ax_left.set_ylim([0, 0.04])
        if i < 3:
            ax_left.set_xticklabels([])
        
        # 保存图例句柄
        for line, label in [(l3, 'L On Signal'),
                           (l4, 'L Off Signal'),
                           (l2, 'Delay Off Signal'),
                           (l1, 'Output')]:
            if label not in added_labels:
                all_handles.append(line)
                all_labels.append(label)
                added_labels.add(label)

        ax_right = axs[i, 1]
        out_line = np.array(cIDP_loc_results[f'{v}']['out'][plot_range[i]:plot_range[i]+40])
        
        l6 = ax_right.plot(x_ticks, cIDP_loc_results[f'{v}']['v_on'][plot_range[i]:plot_range[i]+40], 
                          linestyle='--', color='red', linewidth=1.5)[0]
        l7 = ax_right.plot(x_ticks, cIDP_loc_results[f'{v}']['v_off'][plot_range[i]:plot_range[i]+40], 
                          linestyle='--', color='blue', linewidth=1.5)[0]
        l5 = ax_right.plot(x_ticks, out_line*10, 
                          **OUTPUT_STYLE)[0]
        # l8 = ax_right.plot(cIDP_loc_results[f'{v}']['L_on'][plot_range[i]:plot_range[i]+40], 
        #                   **L_ON_STYLE)[0]
        # l9 = ax_right.plot(cIDP_loc_results[f'{v}']['L_off'][plot_range[i]:plot_range[i]+40], 
        #                   **L_OFF_STYLE)[0]
        
        if i == 0:
            ax_right.set_title('Proposed')
        if i == 3:
            ax_right.set_xlabel('Time (ms)')
        # ax_right.set_ylabel('Response', fontsize=10)
        ax_right.grid(True, alpha=0.3, linestyle='--')
        ax_right.set_ylim([0, 0.35])
        if i < 3:
            ax_left.set_xticklabels([])
        
        # 保存图例句柄
        for line, label in [(l5, 'Output'),
                           (l6, 'V On Signal'),
                           (l7, 'V Off Signal'),
                        #    (l8, 'L On Signal'),
                        #    (l9, 'L Off Signal'),
                           ]:
            if label not in added_labels:
                all_handles.append(line)
                all_labels.append(label)
                added_labels.add(label)

    # 创建统一的底部图例
    fig.legend(all_handles, all_labels,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.00),  # 在底部居中
               ncol=3,  # 5列布局
               frameon=True,
               fancybox=True,
               shadow=False,
               borderpad=0.8,
               labelspacing=0.5,
               handlelength=2.0,
               handletextpad=0.5,
               columnspacing=1.0)

    
    plt.tight_layout(rect=[0, 0.1, 1, 0.98],
                     h_pad=0.2,      # 子图之间的垂直间距
                    )  # 为底部图例留出空间
    plt.show()


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json

def show_results():
    with open(JSON_FILE_PATH, 'r') as f:
        data = json.load(f)
    
    delay_and_correlate_results = data['delay_and_correlate_results']
    cIDP_loc_results = data['cIDP_loc_results']

    # 创建图形和GridSpec
    fig = plt.figure(figsize=(6, 7))
    
    # 创建4行3列的网格，最后一列用于速度坐标轴
    gs = gridspec.GridSpec( 4, 3, 
                            width_ratios=[1, 1, 0.03],  # 第三列窄一些用于速度标签
                            wspace=0.2, 
                            hspace=0.25,
                            top=0.96,    
                            bottom=0.2,
                            left=0.06,  
                            right=0.9)   
    
    # 创建4x2的子图（去掉最后一列）
    axs = []
    for i in range(4):
        row_axes = []
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
        axs.append(row_axes)
    axs = np.array(axs)
    
    v_list = [500, 1000, 2000, 3000]
    plot_range = [385, 193, 96, 64]
    x_ticks = np.arange(40)

    # 统一的线条样式
    L_ON_STYLE = {'linestyle': '-', 'color': 'red', 'linewidth': 1.5}
    L_OFF_STYLE = {'linestyle': '-', 'color': 'blue', 'linewidth': 1.5}
    OUTPUT_STYLE = {'linestyle': '-', 'color': 'black', 'linewidth': 2.5}
    
    # 存储图例句柄和标签
    all_handles = []
    all_labels = []
    added_labels = set()

    for i, v in enumerate(v_list):
        ax_left = axs[i, 0]
        out_line = np.array(delay_and_correlate_results[f'{v}']['out'][plot_range[i]:plot_range[i]+40])
        
        l2 = ax_left.plot(x_ticks, delay_and_correlate_results[f'{v}']['delay_off'][plot_range[i]:plot_range[i]+40], 
                         linestyle='-', color='gray', linewidth=1.5)[0]
        l3 = ax_left.plot(x_ticks, delay_and_correlate_results[f'{v}']['L_on'][plot_range[i]:plot_range[i]+40], 
                         **L_ON_STYLE)[0]
        l4 = ax_left.plot(x_ticks, delay_and_correlate_results[f'{v}']['L_off'][plot_range[i]:plot_range[i]+40], 
                         **L_OFF_STYLE)[0]
        l1 = ax_left.plot(x_ticks, out_line*15, 
                         **OUTPUT_STYLE)[0]
        
        if i == 0:
            ax_left.set_title('Delay and Correlate')
        if i == 3:
            ax_left.set_xlabel('Time (ms)')
        ax_left.set_ylabel('Response')
        ax_left.grid(True, alpha=0.3, linestyle='--')
        ax_left.set_ylim([0, 0.04])
        if i < 3:
            ax_left.set_xticklabels([])
        ax_left.set_yticklabels([])

        
        for line, label in [(l3, r'$L^{+}$ Signal'),
                           (l4, r'$L^{-}$  Signal'),
                           (l1, 'Output'),
                           (l2, r'Delay $L^{-}$  Signal'),
                           ]:
            if label not in added_labels:
                all_handles.append(line)
                all_labels.append(label)
                added_labels.add(label)

        ax_right = axs[i, 1]
        out_line = np.array(cIDP_loc_results[f'{v}']['out'][plot_range[i]:plot_range[i]+40])
        
        l6 = ax_right.plot(x_ticks, cIDP_loc_results[f'{v}']['v_on'][plot_range[i]:plot_range[i]+40], 
                          linestyle='--', color='red', linewidth=1.5)[0]
        l7 = ax_right.plot(x_ticks, cIDP_loc_results[f'{v}']['v_off'][plot_range[i]:plot_range[i]+40], 
                          linestyle='--', color='blue', linewidth=1.5)[0]
        l5 = ax_right.plot(x_ticks, out_line*15, 
                          **OUTPUT_STYLE)[0]
        
        if i == 0:
            ax_right.set_title('Proposed')
        if i == 3:
            ax_right.set_xlabel('Time (ms)')
        ax_right.grid(True, alpha=0.3, linestyle='--')
        ax_right.set_ylim([0, 0.4])
        if i < 3:
            ax_right.set_xticklabels([])
        ax_right.set_yticklabels([])
        
        for line, label in [(l5, 'Output'),
                           (l6, r'$V^{+}$ Signal'),
                           (l7, r'$V^{-}$ Signal')]:
            if label not in added_labels:
                all_handles.append(line)
                all_labels.append(label)
                added_labels.add(label)

    # 创建右侧速度坐标轴
    ax_speed = fig.add_subplot(gs[:, 2])  # 占据所有行的第三列
    pos = ax_speed.get_position()  # 获取当前位置 [left, bottom, width, height]
    new_pos = [pos.x0, pos.y0 + 0.03, pos.width, pos.height * 0.88]  
    ax_speed.set_position(new_pos)
    ax_speed.set_ylabel('Velocity (pixel/s)')
    ax_speed.yaxis.set_label_position("right")
    ax_speed.yaxis.tick_right()
    ax_speed.set_ylim(4, 1)  # 反转y轴，使500在顶部
    
    # 设置刻度位置和标签
    ax_speed.set_yticks([1, 2, 3, 4])
    ax_speed.set_yticklabels(['0.5', '1', '2', '3'])
    
    # 隐藏边框和刻度
    ax_speed.spines['top'].set_visible(False)
    ax_speed.spines['right'].set_visible(True)
    ax_speed.spines['bottom'].set_visible(False)
    ax_speed.spines['left'].set_visible(False)
    ax_speed.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_speed.tick_params(axis='y', which='both', right=True, labelright=True)
    
    # 添加连接线（可选）
    for i in range(4):
        ax_speed.axhline(y=4-i, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    # 创建统一的底部图例
    fig.legend(all_handles, all_labels,
               loc='lower center',
               bbox_to_anchor=(0.43, 0),
               ncol=3,
               frameon=True,
               fancybox=True,
               borderpad=0.8,
               labelspacing=0.5,
               handlelength=2.0,
               handletextpad=0.5,
               columnspacing=1.0)

    plt.savefig(os.path.realpath(__file__).replace('.py', '.png'), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # collect_data()
    show_results()