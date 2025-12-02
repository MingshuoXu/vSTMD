import os
import sys
import math
from collections import deque

import cv2
from cv2 import filter2D, BORDER_CONSTANT
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter



''' LC11 and LC18
reference:
- LC11:
    - Tanaka, R., & Clark, D. A. (2020). Object-displacement-sensitive visual neurons 
        drive freezing in Drosophila. Current Biology, 30(13), 2532-2550.
    - https://github.com/ClarkLabCode/DDModel
- LC18:
    - Klapoetke, N. C., Nern, A., Rogers, E. M., Rubin, G. M., Reiser, M. B., & Card, G. M. (2022). 
        A functionally ordered visual feature map in the Drosophila brain. Neuron, 110(10), 1700-1711.
    - https://zenodo.org/records/5950022
'''


def build_temporal_low_pass_kernel(timeX, tau, fs):
    # MATLAB: ker = timeX .* exp(-timeX/tau)/tau/tau
    ker = timeX * torch.exp(-timeX / tau) / (tau * tau)

    # MATLAB: ker = ker / sqrt(sum(ker.^2)/frameRate);
    ker = ker / torch.sqrt((ker**2).sum() / fs)

    # MATLAB: truncate after 10*tau
    ker = ker[timeX <= tau * 10]

    return ker


def build_temporal_high_pass_kernel(timeX, tau_pre, fs):
    ker = (tau_pre - timeX) * torch.exp(-timeX / tau_pre) / (tau_pre * tau_pre)
    ker = ker / torch.sqrt((ker**2).sum() / fs)
    ker = ker[timeX <= tau_pre * 10]
    return ker


class LC11(nn.Module):
    """
    Online (streaming) version of DDModel that preserves MATLAB's non-recursive FIR convolutions.
    """

    def __init__(self,
                 fs=180,
                 tau_pre=200,
                 kerSizes=(5, 15),
                 kerWeights=(1, 3.5),
                 filterOrder=2,
                 tau_adapt=300,
                 alpha=1,
                 gamma=1000,
                 sigma=10,
                 tau_out=300,
                 device='cpu'):

        self.device = device
        self.fs = fs

        # ---- build time axis like MATLAB ----
        # MATLAB: timeX = linspace(1, snipDuration*1000, snipDuration*frameRate)
        # 对于在线处理，只需要最大 kernel 长度 → 用足够长的 timeX
        Tmax = 500  # ms, 足够覆盖所有 kernel（10*tau）
        timeX = torch.arange(1, Tmax + 1, 1000 / fs).to(device)

        # ---- 0. High-pass kernel ----
        self.kerTpre = build_temporal_high_pass_kernel(timeX, tau_pre, fs)

        # ---- 3. Adaptation kernel ----
        self.kerTadapt = build_temporal_low_pass_kernel(timeX, tau_adapt, fs)

        # ---- 4. Output temporal low-pass kernel ----
        self.kerTout = build_temporal_low_pass_kernel(timeX, tau_out, fs)


        # Time buffers for sliding-window FIR convolution
        self.buffer_S = deque(maxlen=len(self.kerTpre))     # raw input frames
        self.buffer_RCS = deque(maxlen=len(self.kerTadapt))    # after DoG (for adaptation)
        self.buffer_Radapt_conv_Gout = deque(maxlen=len(self.kerTout)) # for output temporal filter

        
        # Build spatial filters (DoG + Gaussian pooling)

        self.DoG = self.build_DoG_kernel(kerSizes, kerWeights).to(device)
        self.spatial_gauss = self.build_2d_gaussian_kernel(sigma).to(device)

        # adaptation parameters
        self.alpha = alpha
        self.gamma = gamma

    # Spatial filters
    def build_DoG_kernel(self, sigs=[5,15], ws=[1,3.5], filter_order=2):
        """
        sigs: [sigma_center, sigma_surround]
        ws: [w_center, w_surround]
        filter_order: 高斯阶数
        """
        # 核大小，3 sigma
        ker_size = int(max(sigs)/5)*5*3
        xs = torch.arange(-ker_size, ker_size+1, 5, device=self.device).float()
        ys = xs.clone()
        x_grid, y_grid = torch.meshgrid(xs, ys, indexing='ij')
        
        gauss1 = torch.exp(-(x_grid**filter_order + y_grid**filter_order)/(2*sigs[0]**filter_order)) / (2*torch.pi*sigs[0]**filter_order)
        gauss2 = torch.exp(-(x_grid**filter_order + y_grid**filter_order)/(2*sigs[1]**filter_order)) / (2*torch.pi*sigs[1]**filter_order)
        
        ker = gauss1*ws[0] - gauss2*ws[1]
        ker /= torch.sqrt((ker**2).sum() * 5 * 5)  # L2 norm
        
        # reshape for conv2d: (out_ch, in_ch, H, W)
        ker = ker.unsqueeze(0).unsqueeze(0)
        return ker


    def build_2d_gaussian_kernel(self, sigma):
        """
        sigma: standard deviation in degrees (与 MATLAB sigma 对应)
        输出: shape (1,1,H,W) 的卷积核，可以直接用于 F.conv2d
        """
        # 核大小，3 sigma，步长假设 5 deg 对应一个像素
        k = int(sigma * 3 / 5) * 5
        xs = torch.arange(-k, k + 1, device=self.device).float()
        ys = xs.clone()
        
        # 生成 2D 高斯
        x_grid, y_grid = torch.meshgrid(xs, ys, indexing='ij')
        ker = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        ker /= ker.sum()  # L1 归一化，也可以改成 L2

        # reshape 成 conv2d 需要的形状 (out_channels, in_channels, H, W)
        ker = ker.unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        return ker

    # Helper: do temporal FIR conv online
    @staticmethod
    def temporal_convolve(buffer, kernel):
        """
        buffer: list of frames (each H×W)
        kernel: 1D vector (len K)

        Output = sum buffer[n-k] * kernel[k]
        """
        K = len(kernel)
        if len(buffer) < K:
            K_use = min(len(buffer), len(kernel))
        else:
            K_use = K

        out = sum(buffer[-K_use+i] * kernel[i] for i in range(K_use))
        return out

    # Single-frame online step
    def forward(self, S):
        S = S.to(self.device)

        # maintain buffer
        self.buffer_S.append(S)

        # 0. Temporal high-pass FIR
        RHP = self.temporal_convolve(self.buffer_S, self.kerTpre)


        # 1. Full-wave rectification
        Rrect = torch.abs(RHP)


        # 2. Center-surround (DoG)
        RCS = F.conv2d(Rrect[None, None],
                       self.DoG, padding='same')[0, 0]

        # update RCS buffer
        self.buffer_RCS.append(RCS)


        # 3. Adaptation: divide by low-pass version
        Q = self.temporal_convolve(self.buffer_RCS, self.kerTadapt)
        Radapt = self.alpha * RCS / (1 + self.gamma * Q)


        # 4. Spatial pooling (Gaussian blur)
        x = Radapt.unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)

        # 边界 pad，用 replicate 模拟 MATLAB tile
        H, W = Radapt.shape
        padH = (self.spatial_gauss.shape[2] - 1) // 2
        padW = (self.spatial_gauss.shape[3] - 1) // 2
        x_padded = F.pad(x, (padW, padW, padH, padH), mode='replicate')

        Radapt_conv_Gout = F.conv2d(x_padded, self.spatial_gauss)  # self.spatial_gauss - shape (1,1,H,W)
        # update Radapt buffer
        self.buffer_Radapt_conv_Gout.append(Radapt_conv_Gout[0,0])
        # Temporal low-pass FIR
        Rout = self.temporal_convolve(self.buffer_Radapt_conv_Gout, self.kerTout)

        return Rout
    
    def process(self, input_frame):
        """
        input_frame: numpy array H×W
        return: lc_out_t (numpy array)
        """
        input_torch = torch.from_numpy(input_frame.astype(np.float32)/255.0).to(self.device)
        lc_out_torch = self.forward(input_torch)
        lc_out = lc_out_torch.cpu().numpy()
        return lc_out


# 1st-order Butterworth IIR Filter (causal) - online version
class IIR1(nn.Module):
    def __init__(self, fc, fs, btype='low'):
        super().__init__()
        # MATLAB butter(1, fc/(fs/2), 'low')
        b, a = butter(1, fc / (fs / 2), btype=btype)

        self.b0 = float(b[0])
        self.b1 = float(b[1])
        self.a1 = float(a[1])

        # previous states
        self.register_buffer("x_prev", None)
        self.register_buffer("y_prev", None)

    def forward(self, x):
        """
        x: tensor H×W
        return: y (same shape)
        """
        if self.x_prev is None:
            self.x_prev = torch.zeros_like(x)
            self.y_prev = torch.zeros_like(x)

        y = self.b0 * x + self.b1 * self.x_prev - self.a1 * self.y_prev

        # update state
        self.x_prev = x.detach()
        self.y_prev = y.detach()

        return y
    

# Difference of Gaussian (static) - fully online
def create_DoG_kernel(sigma, w, ns=4.5, device='cpu'):
    max_val = int(2 * ns * np.ceil(max(sigma)/ns))
    coords = np.arange(-max_val, max_val + ns, ns)
    xx, yy = np.meshgrid(coords, coords)

    Kc = np.exp(-(xx**2 + yy**2)/(2*sigma[0]**2)) / (2*np.pi*sigma[0]**2)

    if sigma[1] > 0:
        Ks = np.exp(-(xx**2 + yy**2)/(2*sigma[1]**2)) / (2*np.pi*sigma[1]**2)
    else:
        Ks = 0

    K = w[0] * Kc - w[1] * Ks
    K = K / np.sqrt(np.sum(K*K) * ns * ns)

    K = torch.tensor(K, dtype=torch.float32, device=device)
    return K[None, None, :, :]


# Saturation (static)
def saturate(x, x_sat, gain):
    out = gain * x
    out = torch.where(x >= x_sat, gain * x_sat, out)
    return out


class LC18(nn.Module):
    def __init__(self, fs=1000, device='cpu'):
        super().__init__()
        self.device = device
        self.param = {
            'hp_fc': 1.0,           # high-pass filter cutoff frequency (Hz)
            'ratio_off2on': 0.416561748,    # OFF to ON response ratio
            'on_adapt_fc': 7.836920311,     # ON adaptation filter cutoff frequency (Hz)
            'on_adapt_scale': 401.3166731,  # ON adaptation scale
            'off_adapt_fc': 7.736672773,    # OFF adaptation filter cutoff frequency (Hz)
            'off_adapt_scale': 372.8078857, # OFF adaptation scale
            'e_on_sat': 0.01,       # ON excitatory saturation level
            'e_on_gain': 1,         # ON excitatory gain
            'i_on_sat': math.inf,   # ON inhibitory saturation level
            'i_on_gain': 1.5,       # ON inhibitory gain
            'e_off_sat': math.inf,  # OFF excitatory saturation level
            'e_off_gain': 1.0,      # OFF excitatory gain
            'i_off_sat': math.inf,          # OFF inhibitory saturation level
            'i_off_gain': 1.250113959,      # OFF inhibitory gain
            'e_on_lp': 19.62960748,         # ON excitatory LP cutoff frequency (Hz)
            'i_on_lp': 18.46903557,         # ON inhibitory LP cutoff frequency (Hz)
            'e_off_lp': 23.4014834,         # OFF excitatory LP cutoff frequency (Hz)
            'i_off_lp': 16.68459067,        # OFF inhibitory LP cutoff frequency (Hz)
            'on_xinh_fc': 2.43798461,       # ON crossover inhibition LP cutoff frequency (Hz)
            'on_xinh_scale': 19.7108115,    # ON crossover inhibition scale
            'off_xinh_fc': 7.846840562,     # OFF crossover inhibition LP cutoff frequency (Hz)
            'off_xinh_scale': 403.7989528,  # OFF crossover inhibition scale
            'w_off': 0.535766147,           # OFF pathway weight
            'lc_scale': 12049,      # LC output scaling factor
            'lc18_fwhm': 3.0,       # LC18 pooling FWHM scaling factor
        }
        self.fs = fs

        # DoG kernels
        self.DoG_contrast = create_DoG_kernel(
            sigma=[4.5, 13.5], w=[1, 3.5], device=device
        )
        self.DoG_pool = create_DoG_kernel(
            sigma=[4.5 * self.param["lc18_fwhm"], 0], w=[1, 0], device=device
        )

        # IIR filters
        self.hp = IIR1(self.param["hp_fc"], fs, btype='high')

        self.on_adapt = IIR1(self.param["on_adapt_fc"], fs)
        self.off_adapt = IIR1(self.param["off_adapt_fc"], fs)

        self.e_on_lp = IIR1(self.param["e_on_lp"], fs)
        self.i_on_lp = IIR1(self.param["i_on_lp"], fs)
        self.e_off_lp = IIR1(self.param["e_off_lp"], fs)
        self.i_off_lp = IIR1(self.param["i_off_lp"], fs)

        self.on_xinh_lp = IIR1(self.param["on_xinh_fc"], fs)
        self.off_xinh_lp = IIR1(self.param["off_xinh_fc"], fs)

        self.ca_lp = IIR1(0.4, fs)


    # process 1 frame
    def forward(self, input_frame):
        """
        input_frame: tensor H×W
        return: lc_out_t, ca_out_t
        """
        ## block 1: photoreceptor processing
        x = input_frame

        ## block 2: contrast
        # (1) high-pass contrast
        b2_tf = self.hp(x)

        # ON/OFF split
        b2_on = torch.relu(b2_tf)
        b2_off = torch.relu(-b2_tf)

        # (2) spatial DoG
        b2_on = F.conv2d(b2_on[None,None], self.DoG_contrast, padding='same')[0,0]
        b2_off = F.conv2d(b2_off[None,None], self.DoG_contrast, padding='same')[0,0]
        b2_off *= self.param["ratio_off2on"]

        # (3) adaptation
        b2_on_adapted = b2_on / (1 + self.param["on_adapt_scale"] * self.on_adapt(b2_on))
        b2_off_adapted = b2_off / (1 + self.param["off_adapt_scale"] * self.off_adapt(b2_off))

        ## block 3: generate combinations of ON,OFF signals
        # (4) saturation + LP (exc/inh)
        b3_e_on  = self.e_on_lp( saturate(b2_on_adapted,  self.param["e_on_sat"],  self.param["e_on_gain"]) )
        b3_i_on  = self.i_on_lp( saturate(b2_on_adapted,  self.param["i_on_sat"],  self.param["i_on_gain"]) )

        b3_e_off = self.e_off_lp( saturate(b2_off_adapted, self.param["e_off_sat"], self.param["e_off_gain"]) )
        b3_i_off = self.i_off_lp( saturate(b2_off_adapted, self.param["i_off_sat"], self.param["i_off_gain"]) )

        # (5) crossover inhibition
        b3_i_on_xinh  = b3_i_on  / (1 + self.param["off_xinh_scale"] * self.off_xinh_lp(b3_e_off))
        b3_i_off_xinh = b3_i_off / (1 + self.param["on_xinh_scale"]  * self.on_xinh_lp(b3_e_on))

        ## block 4: E - I for ON and OFF channels
        # (6) ON/OFF combination
        b4_on  = torch.relu(b3_e_on  - b3_i_on_xinh)
        b4_off = torch.relu(b3_e_off - b3_i_off_xinh)
        b4_full = (1 - self.param["w_off"]) * b4_on + self.param["w_off"] * b4_off

        # (7) LC18 pooling
        S_out = F.conv2d(b4_full[None,None], self.DoG_pool, padding='same')[0,0]

        # (8) Ca LP
        ca = self.ca_lp(torch.relu(self.param["lc_scale"] * S_out))

        return ca
    
    def process(self, input_frame):
        """
        input_frame: numpy array H×W
        return: lc_out_t, ca_out_t (numpy arrays)
        """
        input_torch = torch.from_numpy(input_frame.astype(np.float32)/255.0).to(self.device)
        lc_out_torch = self.forward(input_torch)
        lc_out = lc_out_torch.cpu().numpy()
        return lc_out
    

def test_lc11_on_video():

    vid_number = 290
    vid_path = os.path.join("D:/", "STMD_Dataset", "RIST", f"GX010{vid_number}-1", f"GX010{vid_number}-1.mp4")
    vid_cap = cv2.VideoCapture(vid_path) 

    if not vid_cap.isOpened():
        print("Error: Cannot open video file.")
        sys.exit()

    # Get video resolution
    frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Output", frame_width, frame_height)

    lc11 = LC11(device='cuda', fs=240)

    while True:
        ret, frame = vid_cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_torch = torch.from_numpy(gray_frame.astype(np.float32)/255.0).to('cuda')
        output = lc11.forward(gray_torch)
        if torch.max(output) != 0:
            output /= torch.max(output)
        res = torch.where(output > 0.98)
        y_list = res[0].cpu().numpy()
        x_list = res[1].cpu().numpy()
        if len(res[0]) > 0:
            for y, x in zip(y_list, x_list):
                cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_STAR, 2, 2)

        cv2.imshow("Output", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


def test_lc18_on_video():
    
    vid_number = 290
    vid_path = os.path.join("D:/", "STMD_Dataset", "RIST", f"GX010{vid_number}-1", f"GX010{vid_number}-1.mp4")
    vid_cap = cv2.VideoCapture(vid_path) 

    if not vid_cap.isOpened():
        print("Error: Cannot open video file.")
        sys.exit()

    # Get video resolution
    frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Output", frame_width, frame_height)

    lc18 = LC18(fs=240, device='cuda')

    while True:
        ret, frame = vid_cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_torch = torch.from_numpy(gray_frame.astype(np.float32)/255.0).to('cuda')
        output = lc18.forward(gray_torch)
        if torch.max(output) != 0:
            output /= torch.max(output)
        res = torch.where(output > 0.0001)
        y_list = res[0].cpu().numpy()
        x_list = res[1].cpu().numpy()
        if len(res[0]) > 0:
            for y, x in zip(y_list, x_list):
                cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_STAR, 2, 2)

        cv2.imshow("Output", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    

if __name__ == "__main__":
    test_lc11_on_video()
    # test_lc18_on_video()