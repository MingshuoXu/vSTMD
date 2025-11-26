# demo_vidstream
import os
import sys
import numpy as np
import json

import concurrent.futures
from matplotlib import pyplot as plt
from tqdm import tqdm

# Add the path to the package containing the models
TOP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # code top path
sys.path.append(TOP_PATH)
import config

from smalltargetmotiondetectors.api import inference_task, evaluate_task # type: ignore

FPS = 1000
modelName = 'STMDNet'
V_LIST = range(50,2024,50)
ALPHA_LIST = [i*0.1 for i in range(1, 11)]
G_LEAK_LIST = [i*0.1 for i in range(11)]

def save_as_json(data, file_name='output.json', indent=4):
    """
    Save multiple arguments as a JSON file.

    Parameters:
    - file_name (str): The name of the JSON file to save the data. Defaults to 'output.json'.
    - *args: The data to be saved. Can be multiple objects of any type.
    """
    
    # Ensure the file extension is '.json'
    if not file_name.endswith('.json'):
        file_name += '.json'
    
    # Create the directory path if it does not exist
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save data to JSON file
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=indent)


def STMDNet_task(v, gLeak, alpha=0.3):

    inputpath = os.path.join('D:/STMD_Dataset', 'PanoramaStimuli', 'BV-250-Leftward',
        'SingleTarget-TW-5-TH-5-TV-'+str(v)+'-TL-0-Rightward-Amp-0-Theta-0-TemFre-2-SamFre-'+str(FPS),
        'PanoramaStimuli*.tif')
    
    '''inference'''
    modelOpt, modelDire, _ = inference_task(modelName, inputpath, 'ImgstreamReader', startFrame=1, endFrame=500, gLeak=gLeak, alpha=alpha)
    
    # save
    save_as_json({'modelOpt': modelOpt, 'modelDire': modelDire},
                 os.path.join('D:/', 'STMD_Dataset', 'STMD_OPT',
                            'SingleTarget-TW-5-TH-5-TV-'+str(v)+'-TL-0-Rightward-Amp-0-Theta-0-TemFre-2-SamFre-'+str(FPS),
                            '%s_v%d_alpha%0.1f_gLeak%0.1f_opt.json'%(modelName, v, alpha, gLeak)), 
                 indent = 2,     
                 )


def custom_evaluate_task(v, gLeak, alpha):
    # load groundtruth
    with open(os.path.join(TOP_PATH, 'groundtruth', 
                        'SingleTarget-TW-5-TH-5-TV-'+str(v)+'-TL-0-Rightward-Amp-0-Theta-0-TemFre-2-SamFre-'+str(FPS)+'.json'),
        'r') as file:
        data = json.load(file)
    groundTruth = data['groundTruth']
    # load modelOpt
    with open(os.path.join('D:/', 'STMD_Dataset', 'STMD_OPT',
                            'SingleTarget-TW-5-TH-5-TV-'+str(v)+'-TL-0-Rightward-Amp-0-Theta-0-TemFre-2-SamFre-'+str(FPS),
                            '%s_v%d_alpha%0.1f_gLeak%0.1f_opt.json'%(modelName, v, alpha, gLeak)), 'r') as file:
        data = json.load(file)
    modelOpt = data['modelOpt']
    
    # evaluate
    AUC, AR, AP = evaluate_task(modelOpt, groundTruth, gTError = 3, startFrame=1, endFrame=500, plotFigures=False)
    f1_score = 2*AR*AP/(AR+AP) if AR+AP>0 else 0

    # save
    save_as_json( {'AUC':AUC, 'AR': AR, 'AP': AP, 'F1': f1_score},
                  os.path.join(TOP_PATH, 'result', 
                            'SingleTarget-TW-5-TH-5-TV-'+str(v)+'-TL-0-Rightward-Amp-0-Theta-0-TemFre-2-SamFre-'+str(FPS),
                            '%s_v%d_alpha%0.1f_gLeak%0.1f_result.json'%(modelName, v, alpha, gLeak)), 
                  indent = 4,     
    )

    return AUC, AR, AP, f1_score


def main_compute(max_workers = 8):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for alpha in ALPHA_LIST:
            for v in V_LIST:
                for gLeak in G_LEAK_LIST:
                    future = executor.submit(STMDNet_task, v, gLeak, alpha)
                    futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), 
                           desc='inference task',
                           total=len(V_LIST)*len(ALPHA_LIST)*len(G_LEAK_LIST)
                           ):
            future.result()


def main_evaluate(max_workers = 12):
    aucCurve = [[[None for i in range(len(V_LIST))] for j in range(len(G_LEAK_LIST))] for k in range(len(ALPHA_LIST))]
    arCurve = [[[None for i in range(len(V_LIST))] for j in range(len(G_LEAK_LIST))] for k in range(len(ALPHA_LIST))]
    apCurve = [[[None for i in range(len(V_LIST))] for j in range(len(G_LEAK_LIST))] for k in range(len(ALPHA_LIST))]
    f1Curve = [[[None for i in range(len(V_LIST))] for j in range(len(G_LEAK_LIST))] for k in range(len(ALPHA_LIST))]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for k, alpha in enumerate(ALPHA_LIST):
            for i, v in enumerate(V_LIST):
                for j, gLeak in enumerate(G_LEAK_LIST):
                    future = executor.submit(custom_evaluate_task, v, gLeak, alpha)
                    future.i = i
                    future.j = j
                    future.k = k
                    futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), 
                           desc='inference task',
                           total=len(V_LIST)*len(ALPHA_LIST)*len(G_LEAK_LIST)
                           ):
            _auc, _ar, _ap, _f1_score = future.result()
            aucCurve[future.k][future.j][future.i] = _auc
            arCurve[future.k][future.j][future.i] = _ar
            apCurve[future.k][future.j][future.i] = _ap
            f1Curve[future.k][future.j][future.i] = _f1_score

    save_as_json({'aucCurve': aucCurve, 'arCurve': arCurve, 'apCurve': apCurve, 'f1Curve': f1Curve},
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), f'parameter_analysis.json'),
                 indent = 4,
    )

    print('\nDone...')


def show_result():
    ALPHA_LIST = [i*0.1 for i in range(1, 11)]
    G_LEAK_LIST = [i*0.1 for i in range(11)]

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'parameter_analysis.json'), 'r') as file:
        data = json.load(file)
    aucCurve = data['aucCurve']
    arCurve = data['arCurve']
    apCurve = data['apCurve']
    f1Curve = data['f1Curve']

    aucCurve1 = np.zeros((len(ALPHA_LIST), len(G_LEAK_LIST)))
    arCurve1 = np.zeros((len(ALPHA_LIST), len(G_LEAK_LIST)))
    apCurve1 = np.zeros((len(ALPHA_LIST), len(G_LEAK_LIST)))
    f1Curve1 = np.zeros((len(ALPHA_LIST), len(G_LEAK_LIST)))

    
    for i, curve in enumerate([aucCurve, arCurve, apCurve, f1Curve]):
        for j, alphaData in enumerate(curve):
            for k, gLeakData in enumerate(alphaData):
                if i == 0:
                    aucCurve1[j, k] = np.mean(gLeakData)
                elif i == 1:
                    arCurve1[j, k] = np.mean(gLeakData)
                elif i == 2:
                    apCurve1[j, k] = np.mean(gLeakData)
                elif i == 3:
                    f1Curve1[j, k] = np.mean(gLeakData)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # 画热力图
    for i, data in enumerate([aucCurve1, arCurve1, apCurve1, f1Curve1]):
        j = i//2
        k = i%2
        im = ax[j, k].imshow(data, cmap='hot', interpolation='nearest')
        ax[j, k].set_xticks(np.arange(len(G_LEAK_LIST)))
        ax[j, k].set_yticks(np.arange(len(ALPHA_LIST)))
        ax[j, k].set_xticklabels(['%0.1f' % g for g in G_LEAK_LIST])
        ax[j, k].set_yticklabels(['%0.1f' % al for al in ALPHA_LIST])
        ax[j, k].set_xlabel('gLeak')
        ax[j, k].set_ylabel('alpha')
        for x in range(len(G_LEAK_LIST)):
            for y in range(len(ALPHA_LIST)):
                ax[j, k].text(x, y, '%0.2f'%data[y, x], ha='center', va='center', color='black')

    # 画colorbar
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), orientation='horizontal')
    cbar.set_label('value')

    plt.show()
    

if __name__ == '__main__':
    # main_compute()
    # main_evaluate()
    show_result()



