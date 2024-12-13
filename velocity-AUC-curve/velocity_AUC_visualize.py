import os
import json
import matplotlib.pyplot as plt
from numpy import mean

def visualize():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'velocity_AUC_curve.json'), 'r') as file:
        loaded_data = json.load(file)
        AUCcurveESTMD = loaded_data['AUCcurveESTMD']
        AUCcurveDSTMD = loaded_data['AUCcurveDSTMD']
        AUCcurveFracSTMD = loaded_data['AUCcurveFracSTMD']
        AUCcurveBackbonev2 = loaded_data['AUCcurveBackbonev2']

    V_LIST = range(50,2024,50)
    TAU_LIST = [1,5,11,19,29]

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    for i in range(5):
        mAUC_ESTMD = mean(AUCcurveESTMD[i])
        ax1.plot(V_LIST, AUCcurveESTMD[i], label='tau=%d, mAUC=%0.2f'%(TAU_LIST[i], mAUC_ESTMD))
        mAUC_DSTMD = mean(AUCcurveDSTMD[i])
        ax2.plot(V_LIST, AUCcurveDSTMD[i], label='tau=%d, mAUC=%0.2f'%(TAU_LIST[i], mAUC_DSTMD))
        mAUC_FracSTMD = mean(AUCcurveFracSTMD[i])
        ax3.plot(V_LIST, AUCcurveFracSTMD[i], label='tau=%d, mAUC=%0.2f'%(TAU_LIST[i], mAUC_FracSTMD))
    mAUC_Backbonev2 = mean(AUCcurveBackbonev2)
    ax4.plot(V_LIST, AUCcurveBackbonev2 , 'r-*', linewidth=2, markeredgewidth=2, label='Proposed, mAUC=%0.2f'%mAUC_Backbonev2)

    ax1.set_xlim(0, 2000)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('velocity (pixels/s)')
    ax1.set_ylabel('AUC')
    ax1.title.set_text('ESTMD')
    ax1.legend()

    ax2.set_xlim(0, 2000)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('velocity (pixels/s)')
    ax2.set_ylabel('AUC')
    ax2.title.set_text('DSTMD')
    ax2.legend()

    ax3.set_xlim(0, 2000)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('velocity (pixels/s)')
    ax3.set_ylabel('AUC')
    ax3.title.set_text('FracSTMD')
    ax3.legend()

    ax4.set_xlim(0, 2000)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('velocity (pixels/s)')
    ax4.set_ylabel('AUC')
    ax4.title.set_text('Proposed')
    ax4.legend()
    
    plt.show()

if __name__ == '__main__':
    visualize()