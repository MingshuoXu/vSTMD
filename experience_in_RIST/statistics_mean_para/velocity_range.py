import os
import math

import json
import matplotlib.pyplot as plt
# 全局设置字体
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体家族
plt.rcParams['font.size'] = 13                   # 设置字体大小
plt.rcParams['axes.titlesize'] = 14              # 标题字体大小
plt.rcParams['axes.labelsize'] = 12              # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 11             # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12             # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12             # 图例字体大小
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
import numpy as np


# List of dataset names
datasetList = [
    'GX010071-1', 'GX010220-1', 'GX010228-1', 'GX010230-1', 'GX010231-1',
    'GX010241-1', 'GX010250-1', 'GX010266-1', 'GX010290-1', 'GX010291-1',
    'GX010303-1', 'GX010307-1', 'GX010315-1', 'GX010321-1', 'GX010322-1',
    'GX010327-1', 'GX010335-1', 'GX010336-1', 'GX010337-1'
]

# Path to the dataset directory
datasetPath = os.path.join('D:/STMD_Dataset', 'RIST')

# Function to calculate the projected length of the bbox along the motion vector
def calculate_projectedLength(bbox, motionVector):
    # Extract width and height from the bbox
    width, height = bbox[2], bbox[3]
    
    # Normalize the motion vector to get the unit direction
    motionDirection = np.array(motionVector)
    motionDirection = motionDirection / np.linalg.norm(motionDirection) if np.linalg.norm(motionDirection) > 0 else [0, 0]
    
    if motionDirection[0] == 0:
        projectedLength = width
    elif motionDirection[1] == 0:
        projectedLength = height
    else:
        if width/height > abs(motionDirection[0] / motionDirection[1]):
            projectedLength = height / abs(motionDirection[1])
        else:
            projectedLength = width / abs(motionDirection[0])
    # # Calculate the projection of width and height along the motion direction
    # # The length of the projection is the absolute dot product of the vector (width, 0) and motion direction
    # width_projection = np.abs(np.dot(motionDirection, np.array([width, 0])))
    # height_projection = np.abs(np.dot(motionDirection, np.array([0, height])))
    
    # # The total length along the motion vector is the sum of these projections
    # projectedLength = math.sqrt(width_projection**2 + height_projection**2)
    return projectedLength


def plot_boxplot(data, _title, _ylabel, is_log = True, y_range=None):
    
    plt.figure(figsize=(12, 7))
    
    # widths: 控制小提琴的宽度
    # showmeans: 显示均值点
    # showmedians: 显示中位数线
    parts = plt.violinplot(data, showmeans=True, showmedians=True)

    if is_log:
        plt.yscale('log')

    # 自定义小提琴图的颜色和样式（可选）
    for pc in parts['bodies']:
        pc.set_facecolor('#D4E157')  # 填充颜色
        pc.set_edgecolor('black')     # 边框颜色
        pc.set_alpha(0.7)            # 透明度
    
    # 设置中间线条的颜色（均值、中位数、极值线）
    parts['cmeans'].set_edgecolor('red')    # 均值线设为红色
    parts['cmedians'].set_edgecolor('blue') # 中位数线设为蓝色

    # 设置坐标轴标签
    plt.title(_title)
    plt.ylabel(_ylabel)
    plt.xlabel('Dataset Name')
    if y_range is not None:
        plt.ylim(y_range)
    
    # 设置 X 轴刻度，对应数据集名称
    plt.xticks(range(1, len(datasetList) + 1), datasetList, rotation=45)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(os.path.dirname(__file__), f"{_title}.png"), dpi=300)


def plot_line_chart(datasetName, motionVelocities, projectedLengths, velocity2LengthRatio):
    # Create a figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # 3 rows and 1 column of subplots

        # Plot the motion velocities
        axs[0].plot(motionVelocities, label='Motion Velocities', marker='o')
        axs[0].set_title('Motion Velocities')
        axs[0].set_xlabel('Frame Number')
        axs[0].set_ylabel('Pixel/Frame')  # Set y-axis label
        axs[0].grid(True)
        axs[0].legend()

        # Plot the projected lengths
        axs[1].plot(projectedLengths, label='Projected Lengths', marker='x')
        axs[1].set_title('Projected Lengths')
        axs[1].set_xlabel('Frame Number')
        axs[1].set_ylabel('Pixel')  # Set y-axis label
        axs[1].grid(True)
        axs[1].legend()

        # Plot the ratio of projected length to motion velocity
        axs[2].plot(velocity2LengthRatio, label='Projected Length / Motion Velocity', marker='s')
        axs[2].set_title('Projected Length / Motion Velocity')
        axs[2].set_xlabel('Frame Number')
        axs[2].set_ylabel('Ratio')  # Set y-axis label
        axs[2].grid(True)
        # Add horizontal lines for mean and robustMean
        meanVal = np.mean(velocity2LengthRatio)
        stdVal = np.std(velocity2LengthRatio)
        filteredData = velocity2LengthRatio[(velocity2LengthRatio >= meanVal - 3 * stdVal) & 
                                            (velocity2LengthRatio <= meanVal + 3 * stdVal)]
        robustMean = np.mean(filteredData) / 2
        axs[2].axhline(meanVal, color='red', linestyle='--', label='Mean')  # Red dashed line for mean
        axs[2].axhline(robustMean, color='blue', linestyle='--', label='Robust Mean')  # Blue dashed line for robustMean
        axs[2].legend()

        # Add a title for the entire figure
        fig.suptitle(f'{datasetName}', fontsize=16)

        # Adjust the layout of the subplots
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Leave space at the top for the title
    
def calculate_para():
    # Dictionary to store the results (mean, robustMean) for each dataset
    recordPara = {}
    
    # --- 新增：用于存储每个数据集的 ratio 数组，供箱线图使用 ---
    all_ratios_data = []
    all_velocities_data = []
    all_velocities_data_second = []
    dataset_names = []

    # Iterate through each dataset
    for datasetName in datasetList:
        filePath = os.path.join(datasetPath, datasetName, datasetName + '_annotation.json')
        
        with open(filePath, 'r') as f:
            data = json.load(f)
            
            motionVelocities = []
            projectedLengths = []
            
            for frame_data in data['frames']:
                # 假设每个 frame 只有一个 object
                motionVector = frame_data['objects']['motion_vector']
                bbox = frame_data['objects']['bbox']
                
                if len(motionVector):
                    motionVelocities.append(np.linalg.norm(motionVector))
                    projectedLength = calculate_projectedLength(bbox, motionVector)
                    projectedLengths.append(projectedLength)

        motionVelocities = np.array(motionVelocities)
        projectedLengths = np.array(projectedLengths)

        # Calculate the ratio
        velocity2LengthRatio = np.zeros_like(projectedLengths)
        np.divide(projectedLengths, motionVelocities, out=velocity2LengthRatio, where=motionVelocities>1e-2)

        # --- 新增：保存数据用于后续箱线图 ---
        all_ratios_data.append(velocity2LengthRatio)
        all_velocities_data.append(motionVelocities)
        all_velocities_data_second.append(motionVelocities * 240)  
        dataset_names.append(datasetName)

        recordPara[datasetName] = {'motionVelocities': motionVelocities,
                                    'projectedLengths': projectedLengths,
                                    'velocity2LengthRatio': velocity2LengthRatio
                                    }

    # --- 新增：绘制箱线图 ---
    plot_boxplot(all_velocities_data, "Motion Velocities Distribution Per Frame", "Velocity (pixels/frame)")
    plot_boxplot(all_velocities_data_second, "Motion Velocities Distribution Per Second", "Velocity (pixels/second)", is_log=False)
    plot_boxplot(all_ratios_data, "Velocity to Length Ratio Distribution", "Ratio (frame)", is_log=False, y_range=(0, 100))

    # for key, value in recordPara.items():
    #     plot_line_chart(key, value['motionVelocities'], value['projectedLengths'], value['velocity2LengthRatio'])

    plt.show()


if __name__ == '__main__':
    calculate_para()
    