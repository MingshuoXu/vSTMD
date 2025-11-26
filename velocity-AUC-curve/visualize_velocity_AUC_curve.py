import os
import json
import matplotlib.pyplot as plt
from numpy import mean


def visualize():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'velocity_AUC_curve.json'), 'r') as file:
        loaded_data = json.load(file)
        auc_curve_ESTMD = loaded_data['auc_curve_ESTMD']
        auc_curve_DSTMD = loaded_data['auc_curve_DSTMD']
        auc_curve_FracSTMD = loaded_data['auc_curve_FracSTMD']
        auc_curve_vSTMD = loaded_data['auc_curve_vSTMD']
        auc_curve_vSTMD_F = loaded_data['auc_curve_vSTMD_F']
        V_LIST = loaded_data['v_list']
        TAU_LIST = loaded_data['tau_list']

    fig, axs = plt.subplots(2, 2)

    for i in range(len(TAU_LIST)):
        mAUC_ESTMD = mean(auc_curve_ESTMD[i])
        axs[0, 0].plot(V_LIST, auc_curve_ESTMD[i], color=f"C{i}", label='tau=%d, mAUC=%0.2f'%(TAU_LIST[i], mAUC_ESTMD))
        mAUC_DSTMD = mean(auc_curve_DSTMD[i])
        axs[0, 1].plot(V_LIST, auc_curve_DSTMD[i], color=f"C{i}", label='tau=%d, mAUC=%0.2f'%(TAU_LIST[i], mAUC_DSTMD))
        mAUC_FracSTMD = mean(auc_curve_FracSTMD[i])
        axs[1, 0].plot(V_LIST, auc_curve_FracSTMD[i], color=f"C{i}", label='tau=%d, mAUC=%0.2f'%(TAU_LIST[i], mAUC_FracSTMD))
    mAUC_vSTMD = mean(auc_curve_vSTMD)
    axs[1, 1].plot(V_LIST, auc_curve_vSTMD , color=f"C{i+1}", linewidth=2, markeredgewidth=2, label='vSTMD')
    mAUC_vSTMD_F = mean(auc_curve_vSTMD_F)
    axs[1, 1].plot(V_LIST, auc_curve_vSTMD_F , color=f"C{i+2}", linewidth=2, markeredgewidth=2, label='vSTMD_F')

    x_axis_max_range = 3000

    for i in range(4):
        axs[i//2, i%2].set_xlim(0, x_axis_max_range)
        axs[i//2, i%2].set_ylim(0, 1)
        # axs[i//2, i%2].set_xlabel('velocity (pixels/s)')
        # axs[i//2, i%2].set_ylabel('AUC') 
        # axs[i//2, i%2].legend()

    axs[0, 0].title.set_text('ESTMD')
    axs[0, 1].title.set_text('DSTMD')
    axs[1, 0].title.set_text('FracSTMD')
    axs[1, 1].title.set_text('Proposed')
    axs[1, 1].legend()

    # 设置图例
    leg_list = [f'tau={tau}' for tau in TAU_LIST] + ['without Tau']
    handles = [plt.Line2D([0], [0], color=f"C{i}", lw=2) for i in range(len(leg_list))]
    plt.figlegend(handles, leg_list,  
               fontsize=11, title_fontsize=14,
               bbox_to_anchor=(0.05, -0.02, 0.9, 0.1),
               loc='lower center', ncol=3,
               handlelength=1.5,  # 图例线条长度
               fancybox=True,  # 圆角边框
               framealpha=0.9, # 透明度
            )
    plt.tight_layout()
    plt.subplots_adjust(top = 0.88, bottom=0.2)
    

    plt.savefig(f'{os.path.realpath(__file__)[:-3]}.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    visualize()