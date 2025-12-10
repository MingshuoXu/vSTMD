import os
import sys
import time
import torch

# DEVICE = 'cpu' # 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get the full path of this file
filePath = os.path.realpath(__file__)
# Find the index of '/+smalltargetmotiondetectors/'
indexPath = filePath.rfind('OneDrive')
# Add the path to the package containing the models
STMD_PYTHON_PATH = os.path.join(filePath[:indexPath], 'OneDrive', '1_Code', '0_GitHub',
                                 'Small-Target-Motion-Detectors', 'python')
sys.path.append(STMD_PYTHON_PATH)

from smalltargetmotiondetectors.api import (instancing_model, get_visualize_handle, inference) # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader, ImgstreamReader # type: ignore
from smalltargetmotiondetectors.util.compute_module import matrix_to_sparse_list # type: ignore


def run_and_record(videoPath: str = None, saveName: str = None, imgPath: str = None):

    ''' Model instantiation '''
    objModel = instancing_model('vSTMD_F', device=DEVICE)


    ''' Input '''
    if videoPath is not None:
        hSteam = VidstreamReader(videoPath)
    elif imgPath is not None:
        hSteam = ImgstreamReader(imgPath)


    ''' Get visualization handle '''
    hVisual = get_visualize_handle(objModel.__class__.__name__)
    hVisual.isSaveAsVideo = True
    hVisual.savePath = os.path.dirname(filePath)
    hVisual.videoName = saveName


    ''' Initialize the model '''
    # set the parameter list
    objModel.set_para()
    # print the parameter list
    objModel.print_para()
    # init
    objModel.init_config()


    totalTime = 0
    '''Run inference'''
    while hSteam.hasFrame and hVisual.hasFigHandle:

        # Get the next frame from the input source
        grayImg, colorImg = hSteam.get_next_frame()

        if DEVICE == 'cuda':
            grayImg = torch.from_numpy(grayImg).to(device=DEVICE).float().unsqueeze(0).unsqueeze(0)
        
        # Perform inference using the model
        result, runTime = inference(objModel, grayImg)
        totalTime += runTime
        # result['response'] = matrix_to_sparse_list(result['response'])


        # # direction
        # direction  = result['direction']
        # if (direction is not None) and len(direction) and len(result['response']):
        #     directionListType = [[y, x, direction[x, y]] for y, x, _ in result['response']]
        # else:
        #     directionListType = []
        # result['direction'] = directionListType
        
        # Visualize the result
        if DEVICE == 'cuda':
            result = {k: v.cpu().numpy().squeeze(0).squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        hVisual.show_result(colorImg, result, runTime)

    print(f"Total time: {totalTime:.4f} seconds")


if __name__ == '__main__':

    # vidPth = os.path.join('D:', 'STMD_Dataset', 'RIST', 'GX010290-1', 'GX010290-1.mp4')
    # saveNme = 'vSTMD_F_RIST_GX010290-1.mp4'

    # vidPth = os.path.join('D:', 'STMD_Dataset', 'Real-World-Scence-Material', 'From_Dr_YiZheng', 'fly_bird_and_fish.mp4')
    # saveNme = 'vSTMD_F_flying_bird.mp4'

    imgPath = os.path.join(STMD_PYTHON_PATH, '..', 'demodata', 'imgstream', 'DemoFig*.jpg')
    saveNme = 'vSTMD_F_butterfly.mp4'

    # run_and_record(videoPath=vidPth, saveName=saveNme)
    run_and_record(imgPath=imgPath, saveName=saveNme)
