import os
ITEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ITEM_DIR)
import time

import torch
import cv2

# DEVICE = 'cpu' # 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Add the path to the package containing the models
STMD_PYTHON_PATH = os.path.join('D:/', '11_Code', 'Small-Target-Motion-Detectors', 'python')
sys.path.append(STMD_PYTHON_PATH)
from smalltargetmotiondetectors.api import (instancing_model, inference) # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader, ImgstreamReader # type: ignore

from utils import show_result_in_cv2, FrameIterator


def inference_and_show(model, sequence_iterator, visulize_name = 'Result', show_number=10):

    total_time = 0
    '''Run inference'''
    while True:

        # Get the next frame from the input source
        gray_img, color_img, cap = sequence_iterator.get_next_frame()
        if gray_img is None:
            break

        if DEVICE == 'cuda':
            gray_img = torch.from_numpy(gray_img).to(device=DEVICE).float().unsqueeze(0).unsqueeze(0)
        
        # Perform inference using the model
        result, run_time = inference(model, gray_img)
        total_time += run_time
        
        
        cap = show_result_in_cv2(visulize_name, color_img, result, show_number=show_number, device=DEVICE)

        if cap is False:
            break

    print(f"Total processing time: {total_time:.4f} seconds")


def main():
    ''' Model instantiation '''
    model = instancing_model('vSTMD_F', device=DEVICE)
    ''' Initialize the model '''
    # set the parameter list
    model.set_para()
    # print the parameter list
    model.print_para()
    # init
    model.init_config()

    # sequence_iterator = FrameIterator(os.path.join('D:/', 'STMD_Dataset', 'RIST', 'GX010290-1', 'GX010290-1.mp4'), is_video=True)
    sequence_iterator = FrameIterator(os.path.join('D:/', 'STMD_Dataset', 'XS-VID', 'images', 'UAVTOD_DJI0824Part5_0'), is_video=False)
    
    ''' Get visualization handle '''
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', sequence_iterator.img_width, sequence_iterator.img_height)

    inference_and_show(model, sequence_iterator, show_number=10)


if __name__ == '__main__':
    main()

