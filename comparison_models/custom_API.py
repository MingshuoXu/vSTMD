import os
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(CURR_PATH)

import argparse
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
import cv2
import numpy as np
# from PIL import Image
# from matplotlib import pyplot as plt
import time
import ptlflow

from vSTMD.vSTMD import vSTMD_gpu


def flow_to_ang(flow):
    # Convert flow to theta for orientation
    fx, fy = flow[:,:,0], flow[:,:,1]
    
    ang = np.arctan2(-fy, fx)
    ang[ang < 0] += 2 * np.pi  # Ensure angles are in [0, 2*pi]
    return ang


def img2tensor(frame):
    """Convert BGR OpenCV frame to normalized RGB tensor on GPU."""
    # BGR to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize [0, 255] -> [0, 1]
    frame_tensor = torch.from_numpy(
        np.ascontiguousarray(frame_rgb)  # 确保内存连续
    ).permute(2, 0, 1).float()
    
    # Add batch dimension and move to GPU
    return frame_tensor


def flow_to_image(flow):
    # Convert flow to HSV image for visualization
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def prepared_for_rist(ipt):
    """Prepare frame for RIST dataset format."""
    # Convert BGR to RGB
    frame = np.pad(
            ipt, 
            pad_width=((0, 2), (0, 0), (0, 0)),  # 仅在第一维末尾补 pad_rows 行
            mode='constant', 
            constant_values=0  # 填充值为 0
        )
        
    # Convert frame to tensor
    frame_tensor = img2tensor(frame)
    
    return frame_tensor


class CustomFlowDiffuser():
    
    def __init__(self):
        # Initialize model
        # sys.path.append(os.path.join(CURR_PATH, "FlowDiffuser"))
        from FlowDiffuser.core.flowdiffuser import FlowDiffuser # type: ignore

        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--dataset', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', default=True, help='use small model (default: True)')
        parser.add_argument('--mixed_precision', action='store_true', default=True, help='use mixed precision (default: True)')
        parser.add_argument('--alternate_corr', action='store_true', default=True, help='use efficient correlation implementation (default: True)')
        args = parser.parse_args()

        weight_path = os.path.join(CURR_PATH, "FlowDiffuser", "weights", "FlowDiffuser-things.pth")

        self.model = torch.nn.DataParallel(FlowDiffuser(args))
        self.model.load_state_dict(torch.load(weight_path))
        self.model.cuda()
        self.model.eval()

        self.lastImg_cuda = None
        self.iters=24

    def process(self, newFrame):
        with torch.no_grad():
            if self.lastImg_cuda is None:
                # First frame, no previous frame to compare
                self.lastImg_cuda = newFrame[None].cuda()
                return None
        
        
            self.currImg_cuda = newFrame[None].cuda()

            _, flow_pr = self.model(self.lastImg_cuda, 
                                    self.currImg_cuda, 
                                    iters=self.iters, 
                                    test_mode=True)
        
            # Update last image
            self.lastImg_cuda = self.currImg_cuda
        
        flow_cpu = flow_pr[0].cpu().permute(1, 2, 0).detach().numpy()
        return flow_cpu
    

class CustomRAFT():
    
    def __init__(self):
        # Initialize model

        # sys.path.append(os.path.join(CURR_PATH, "RAFT"))
        from RAFT.core.raft import RAFT # type: ignore
        from RAFT.core.utils.utils import InputPadder # type: ignore
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', default=True, help='use small model (default: True)')
        parser.add_argument('--mixed_precision', action='store_true', default=True, help='use mixed precision (default: True)')
        parser.add_argument('--alternate_corr', action='store_true', default=True, help='use efficient correlation implementation (default: True)')
        args = parser.parse_args()

        weight_path = os.path.join(CURR_PATH, "RAFT", "models", "raft-small.pth")

        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(weight_path))
        self.model.cuda()
        self.model.eval()

        self.lastImg_cuda = None

        self.InputPadder = InputPadder
        

    def process(self, newFrame):

        with torch.no_grad():
            if self.lastImg_cuda is None:
                # First frame, no previous frame to compare
                self.lastImg_cuda = newFrame[None].cuda()
                return None
            

            self.currImg_cuda = newFrame[None].cuda()

            padder = self.InputPadder(self.lastImg_cuda.shape)
            image1, image2 = padder.pad(self.lastImg_cuda, self.currImg_cuda)

            _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
            
            # Update last image
            self.lastImg_cuda = self.currImg_cuda
        
        flow_cpu = flow_up[0].cpu().permute(1, 2, 0).detach().numpy()
        return flow_cpu


class CustomSEA_RAFT():

    def __json_to_args(self, json_path):
        # return a argparse.Namespace object
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        args = argparse.Namespace()
        args_dict = args.__dict__
        for key, value in data.items():
            args_dict[key] = value
        return args

    def __parse_args(self, parser):
        entry = parser.parse_args()
        json_path = os.path.join(CURR_PATH, "SEA_RAFT", "config", "eval", "spring-M.json")
        args = self.__json_to_args(json_path)
        args_dict = args.__dict__
        for index, (key, value) in enumerate(vars(entry).items()):
            args_dict[key] = value
        return args
    
    def __init__(self):
        # Initialize model

        # sys.path.append(os.path.join(CURR_PATH, "SEA_RAFT"))
        from SEA_RAFT.core.raft import RAFT # type: ignore
        from SEA_RAFT.core.utils.utils import load_ckpt # type: ignore
        import torch.nn.functional as F
        
        parser = argparse.ArgumentParser()
        # parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
        # parser.add_argument('--model', help='checkpoint path', required=True, type=str)
        # parser.add_argument('')
        self.args = self.__parse_args(parser)


        weight_path = os.path.join(CURR_PATH, "SEA_RAFT", "models", "Tartan-C-T-TSKH-spring540x960-M.pth")
        self.model = RAFT(self.args)
        load_ckpt(self.model, weight_path)
        self.model = self.model.cuda()
        self.model.eval()

        self.F = F


        self.lastImg_cuda = None

    def process(self, newFrame):
        with torch.no_grad():
            if self.lastImg_cuda is None:
                # First frame, no previous frame to compare
                self.lastImg_cuda = newFrame[None].cuda()
                return None
            
            self.currImg_cuda = newFrame[None].cuda()

            img1 = self.F.interpolate(self.lastImg_cuda, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
            img2 = self.F.interpolate(self.currImg_cuda, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
            H, W = img1.shape[2:]

            output = self.model(img1, img2, iters=self.args.iters, test_mode=True)
            flow = output['flow'][-1]
            # info = output['info'][-1]

            flow_down = self.F.interpolate(flow, scale_factor=0.5 ** self.args.scale, mode='bilinear', align_corners=False) * (0.5 ** self.args.scale)
            _ = self.F.interpolate(output['info'][-1], scale_factor=0.5 ** self.args.scale, mode='area')
            
            # Update last image
            self.lastImg_cuda = self.currImg_cuda
        
        flow_cpu = flow_down[0].cpu().permute(1, 2, 0).detach().numpy()
        return flow_cpu

    
class CustomMemFlow():
    
    def __init__(self):
        # Initialize model
        import random

        # sys.path.append(os.path.join(CURR_PATH, "MemFlow"))
        from MemFlow.core.Networks import build_network # type: ignore
        from MemFlow.core.utils.utils import forward_interpolate # type: ignore
        from MemFlow.inference import inference_core_skflow as inference_core # type: ignore
        from MemFlow.configs.sintel_memflownet_t import get_cfg # type: ignore

        parser = argparse.ArgumentParser()
        parser.add_argument('--name', default='MemFlowNet', choices=['MemFlowNet', 'MemFlowNet_T'], help="name your experiment")
        parser.add_argument('--stage', help="determines which dataset to use for training")
        parser.add_argument('--restore_ckpt', help="restore checkpoint")
        parser.add_argument('--seq_dir', default='default')
        parser.add_argument('--vis_dir', default='default')
        args = parser.parse_args()

        self.cfg = get_cfg()
        self.cfg.update(vars(args))

        # initialize random seed
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        np.random.seed(1234)
        random.seed(1234)

        
        self.forward_interpolate = forward_interpolate

        model = build_network(self.cfg).cuda()

        weight_path = os.path.join(CURR_PATH, "MemFlow", "ckpts", "MemFlowNet_T_sintel.pth")
        ckpt = torch.load(weight_path, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            for key in ckpt_model.keys():
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            model.load_state_dict(ckpt_model, strict=True)
        else:
            model.load_state_dict(ckpt_model, strict=True)

        model.eval()

        self.processor = inference_core.InferenceCore(model, config=self.cfg)

        self.lastImg_cuda = None


    def process(self, newFrame):
        with torch.no_grad():
            if self.lastImg_cuda is None:
                # First frame, no previous frame to compare
                self.lastImg_cuda = newFrame.cuda()
                self.flow_prev = None
                return None
        
        
            self.currImg_cuda = newFrame.cuda()
            inputImgs = torch.stack([self.lastImg_cuda, self.currImg_cuda]).cuda().unsqueeze(0)

            flow_low, flow_pre = self.processor.step(inputImgs, 
                                add_pe=('rope' in self.cfg and self.cfg.rope), 
                                flow_init=self.flow_prev)
            self.flow_prev = self.forward_interpolate(flow_low[0])[None].cuda()
            del inputImgs, flow_low

        
            # Update last image
            self.lastImg_cuda = self.currImg_cuda
        
        flow_cpu = flow_pre[0].cpu().permute(1, 2, 0).detach().numpy()
        return flow_cpu
    

class CustomStreamFlow():
    
    def __init__(self):
        # Initialize model
        # sys.path.append(os.path.join(CURR_PATH, "StreamFlow"))
        from StreamFlow.demo import StreamFlowT4 # type: ignore

        weights = os.path.join(CURR_PATH, "StreamFlow", "weights", "streamflow-spring.pth")
        self.model = StreamFlowT4(weights).to('cuda')

        self.frameList = []  # Store last 3 frames
        

    def process(self, newFrame):
        with torch.no_grad():
            self.frameList.append(newFrame.cuda())
            if len(self.frameList) < 4:
                # First frame, no previous frames to compare
                return None
        
            flows = self.model(torch.stack(self.frameList,dim=0).unsqueeze(0))
            del self.frameList[0:3]  # Remove the oldest frame

        flow_cpu = [flow[0].cpu().permute(1, 2, 0).detach().numpy() for flow in flows]
        return flow_cpu
    

class CustomDpFlow():
    
    def __init__(self):
        # Initialize model
        self.model = ptlflow.get_model('dpflow', ckpt_path='things')
        self.model.eval()
        self.model.cuda()

        self.lastImg_cuda = None

    def process(self, newFrame):
        with torch.no_grad():
            if self.lastImg_cuda is None:
                # First frame, no previous frame to compare
                self.lastImg_cuda = newFrame.cuda()
                return None
        
            self.currImg_cuda = newFrame.cuda()

            # inputs is a dict {'images': torch.Tensor}
            # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
            # (1, 2, 3, H, W)
            inputs = {'images': torch.stack([self.lastImg_cuda, self.currImg_cuda]).unsqueeze(0),
                      }

            # Forward the inputs through the model
            predictions = self.model(inputs)

            # The output is a dict with possibly several keys,
            # but it should always store the optical flow prediction in a key called 'flows'.
            flows = predictions['flows']

            # flows will be a 5D tensor BNCHW.
            # This example should print a shape (1, 1, 2, H, W).
            # print(flows.shape)

            # Update last image
            self.lastImg_cuda = self.currImg_cuda

        flow_cpu = flows[0,0].cpu().permute(1, 2, 0).detach().numpy()
        return flow_cpu
    

class CustomModelByPtlFlow():
    # This is a custom template for using any model from ptlflow
    def __init__(self, modelName='dpflow', ckpt_path='things'):
        # Initialize model
        self.model = ptlflow.get_model(modelName, ckpt_path)
        self.model.eval()
        self.model.cuda()

        self.lastImg_cuda = None

    def process(self, newFrame):
        with torch.no_grad():
            if self.lastImg_cuda is None:
                # First frame, no previous frame to compare
                self.lastImg_cuda = newFrame.cuda()
                return None
        
            self.currImg_cuda = newFrame.cuda()

            # inputs is a dict {'images': torch.Tensor}
            # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
            # (1, 2, 3, H, W)
            inputs = {'images': torch.stack([self.lastImg_cuda, self.currImg_cuda]).unsqueeze(0),
                      }

            # Forward the inputs through the model
            predictions = self.model(inputs)

            # The output is a dict with possibly several keys,
            # but it should always store the optical flow prediction in a key called 'flows'.
            flows = predictions['flows']

            # flows will be a 5D tensor BNCHW.
            # This example should print a shape (1, 1, 2, H, W).
            # print(flows.shape)

            # Update last image
            self.lastImg_cuda = self.currImg_cuda

        flow_cpu = flows[0,0].cpu().permute(1, 2, 0).detach().numpy()
        return flow_cpu


def test_model(hModel, vidpath=None, isVisulize=True):
    cap = cv2.VideoCapture(vidpath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
        
    timeTic = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        frame = np.pad(
            frame, 
            pad_width=((0, 2), (0, 0), (0, 0)),  # 仅在第一维末尾补 pad_rows 行
            mode='constant', 
            constant_values=0  # 填充值为 0
        )
        
        # Convert frame to tensor
        frame_tensor = img2tensor(frame)
        
        # Process the frame
        flow = hModel.process(frame_tensor)
        
        if flow is not None:
            # flow_ang = flow_to_ang(flow)
            print(f"Processed frame in {time.time() - timeTic:.2f} seconds")
            timeTic = time.time()
            if isVisulize:
                flow_vis = flow_to_image(flow)
                combined = np.hstack([frame, flow_vis])
                cv2.imshow('Input | Optical Flow', combined)

        if isVisulize:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    vidPath = os.path.join('D:/', 'STMD_Dataset', 'RIST', 'GX010290-1', 'GX010290-1.mp4')
    
    
    hModel = CustomRAFT()
    # hModel = CustomSEA_RAFT()

    # hModel = CustomMemFlow()

    # hModel = CustomStreamFlow()

    # hModel = CustomDpFlow()

    # hModel = vSTMD_gpu()
    # hModel.init_config()
    
    # test_model(hModel, vidPath, False)
    test_model(hModel, vidPath)


