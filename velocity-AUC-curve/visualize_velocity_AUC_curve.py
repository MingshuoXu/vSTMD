import os
import json
import matplotlib.pyplot as plt
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



def visualize():
    # 配置常量
    FIG_SIZE = (6, 5)
    LINE_WIDTH = 2
    Y_LIM = (0, 1)
    X_LIM = (0.1, 10)

    
    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'velocity_AUC_curve.json')
    
    with open(data_path, 'r') as file:
        data = json.load(file)
        auc_curve_ESTMD = data['auc_curve_ESTMD']
        auc_curve_DSTMD = data['auc_curve_DSTMD']
        auc_curve_FeedbackSTMD = data['auc_curve_FeedbackSTMD']
        auc_curve_vSTMD = data['auc_curve_vSTMD']
        auc_curve_vSTMD_F = data['auc_curve_vSTMD_F']
        V_LIST = [v/1000 for v in data['v_list']]
        TAU_LIST = data['tau_list']
    
    # 创建图形
    fig, axs = plt.subplots(2, 2, figsize=FIG_SIZE)

    colors = plt.cm.Set2.colors  # 使用Set2色彩映射
    
    # 前三个子图
    for i in range(len(TAU_LIST)):
        # 使用更柔和的颜色
        color = colors[i % len(colors)]
        
        axs[0, 0].plot(V_LIST, auc_curve_ESTMD[i], 
                      color=color,
                      linewidth=1.5,)
        
        axs[0, 1].plot(V_LIST, auc_curve_DSTMD[i], 
                      color=color,
                      linewidth=1.5,)
        
        axs[1, 0].plot(V_LIST, auc_curve_FeedbackSTMD[i], 
                      color=color,
                      linewidth=1.5,)
    
    # 绘制最后一个子图  

    axs[1, 1].plot(V_LIST, auc_curve_vSTMD, 
                   color='red',
                   linewidth=2,      # 更粗
                   solid_capstyle='round',
                   label='vSTMD')
    
    axs[1, 1].plot(V_LIST, auc_curve_vSTMD_F, 
                   color='blue',
                   linewidth=2,
                   solid_capstyle='round',
                   label='vSTMD_F')

    
    # 配置子图
    titles = ['Elementary delay and correlate', 
              'Directional delay and correlate',
              'Feedback facilitated', 
              r'Proposed (without $\tau$)']
    
    for idx, ax in enumerate(axs.flat):
        ax.set_ylim(*Y_LIM)
        ax.set_xscale('log')
        ax.set_xlim(*X_LIM)
        ax.set_title(titles[idx], pad=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # 设置标签
        if idx >= 2:  # 最后一行
            ax.set_xlabel('velocity (pixel / frame)')
        if idx % 2 == 0:  # 第一列
            ax.set_ylabel('AUC')

    
    # 添加子图标签
    labels = ['(A)', '(B)', '(C)', '(D)']
    for idx, ax in enumerate(axs.flat):
        # 在左上角添加标签
        ax.text(-0.25, 1.27, labels[idx], 
                transform=ax.transAxes,  # 使用坐标轴相对坐标
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='left')
    
    # 创建全局图例
    legend_labels = [r'$\tau$'+f' = {tau}' for tau in TAU_LIST] + ['vSTMD', 'vSTMD-F']
    legend_handles = []
    
    for i in range(len(TAU_LIST)):
        legend_handles.append(plt.Line2D([0], [0], 
                                        color=colors[i % len(colors)], 
                                        linewidth=LINE_WIDTH))
    
    legend_handles.append(plt.Line2D([0], [0], 
                                    color='red',
                                    linewidth=LINE_WIDTH))
    legend_handles.append(plt.Line2D([0], [0], 
                                    color='blue',
                                    linewidth=LINE_WIDTH))
    
    # 添加全局图例
    fig.legend(legend_handles, legend_labels,
               bbox_to_anchor=(0.5, 0.13),
               loc='upper center',
               ncol=4,
               frameon=True,
               framealpha=0.9)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # 为底部图例留出空间
    
    # 保存和显示
    output_path = f'{os.path.splitext(os.path.realpath(__file__))[0]}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    visualize()