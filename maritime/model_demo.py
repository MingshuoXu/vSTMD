import os
ITEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ITEM_DIR)
import torch

import cv2
import numpy as np
from tqdm import tqdm

# DEVICE = 'cpu' # 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STMD_PYTHON_PATH = os.path.join('D:/', '11_Code', 'Small-Target-Motion-Detectors', 'python')
sys.path.append(STMD_PYTHON_PATH)
from smalltargetmotiondetectors.api import (instancing_model, inference) # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader # type: ignore
from utils import FrameIterator, nms


video_map = {
        'demo1': {'name': 'demo1-SeaDronesSee-696-1410.mp4', 
                  'start_frame': 696, 'end_frame': 1410},
        'demo2': {'name': 'demo2-SeaDronesSee-1697-2411.mp4',
                    'start_frame': 1697, 'end_frame': 2411},
        'demo3': {'name': 'demo3-SeaDronesSee-3666-4166.mp4',
                    'start_frame': 3666, 'end_frame': 4166},
        'demo4': {'name': 'demo4-SeaDronesSee-22931-23545.mp4',
                    'start_frame': 22931, 'end_frame': 23545},
        'demo5': {'name': 'demo5-SeaDronesSee-29713-30312.mp4',
                    'start_frame': 29713, 'end_frame': 30312},
    }


def get_top_k_torch(response_tensor, k=1000):
    """
    输入: response_tensor: (H, W) 的 torch.Tensor (可以是 CUDA)
    输出: list of (y, x, score)
    """
    # 1. 获取形状
    H, W = response_tensor.shape[-2:]
    
    # 防止 k 超过像素总数
    k = min(k, H * W)

    # 2. Flatten (展平)
    flat_response = response_tensor.view(-1)

    # 3. TopK 核心操作
    # torch.topk 默认就是降序 (largest=True)，且只找前 k 个，速度极快
    top_n_values, top_n_indices = torch.topk(flat_response, k=k)

    # 4. Unravel Index (计算坐标)
    # PyTorch 旧版本没有 np.unravel_index，用除法和取模即可
    # y = index // width
    # x = index % width
    top_n_y = torch.div(top_n_indices, W, rounding_mode='floor') # y (row)
    top_n_x = top_n_indices % W                                  # x (col)

    # 5. 格式转换 (适配之前的 evaluator)
    # 如果数据在 GPU 上，必须先 .cpu() 才能转 list
    if response_tensor.is_cuda:
        top_n_y = top_n_y.cpu()
        top_n_x = top_n_x.cpu()
        top_n_values = top_n_values.cpu()

    # 转换为 Python list of tuples: [(y, x, val), ...]
    # 使用 .numpy().tolist() 或者直接 .tolist()
    targets = list(zip(top_n_y.tolist(), top_n_x.tolist(), top_n_values.tolist()))
    
    return targets


def inference_video(sequence_iterator, start_frame, end_frame, is_save, show_number=10):
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    if sequence_iterator.img_width > 1920:
        scale_factor = 1920 / sequence_iterator.img_width
        new_width = 1920
        new_height = int(sequence_iterator.img_height * scale_factor)
        cv2.resizeWindow('Result', new_width, new_height)

    # 1. 准备视频写入器 (重命名为 video_writer 以避免冲突)
    video_writer = None
    if is_save:
        save_dir = os.path.join(ITEM_DIR, 'maritime', 'results')
        os.makedirs(save_dir, exist_ok=True) # 确保目录存在
        save_path = os.path.join(save_dir, f'inference_output_{start_frame}_{end_frame}.mp4')
        
        # 注意: sequence_iterator.img_width 需确保存在，否则用 color_img.shape 获取
        video_writer = cv2.VideoWriter(
            save_path, 
            cv2.VideoWriter_fourcc(*'mp4v'),
            30, 
            (sequence_iterator.img_width, sequence_iterator.img_height)
        )
        print(f"Video will be saved to: {save_path}")

    ''' Model instantiation '''
    objModel = instancing_model('vSTMD_F_L', device=DEVICE) 
    objModel.set_para()
    objModel.print_para()
    objModel.init_config()

    totalTime = 0
    video_preds = [] # 用于收集所有帧的预测结果，传给评估器使用

    '''Run inference'''
    # 使用 tqdm 显示进度
    print(f"Start Inference from frame {start_frame} to {end_frame}")
    
    # 确保 iterator 定位到 start_frame (取决于你的 iterator 实现，这里假设它从头开始或已同步)
    # 如果 sequence_iterator 是基于索引的，确保这里逻辑对齐
    
    for frame_idx in tqdm(range(start_frame, end_frame)):

        # Get the next frame
        # 修正: 不要用 cap 接收返回值，避免覆盖 video_writer
        gray_img, color_img, status = sequence_iterator.get_next_frame()
        
        if gray_img is None:
            print("End of video stream.")
            break

        # Pre-process
        if DEVICE == 'cuda':
            # 保持维度 (1, 1, H, W) 用于输入模型
            input_tensor = torch.from_numpy(gray_img).to(device=DEVICE).float().unsqueeze(0).unsqueeze(0)
        else:
            input_tensor = gray_img # 根据 CPU 模型需求调整
        
        # Inference
        result, runTime = inference(objModel, input_tensor)
        totalTime += runTime

        # Post-process & Visualization
        targets = [] # 当前帧的预测结果 [(y, x, score), ...]

        if DEVICE == 'cuda':
            # --- GPU 处理流程 ---
            # 1. 归一化 (In-place 可能会有问题，建议拷贝或确保安全)
            max_response = torch.max(result['response']) 
            if max_response > 0:
                result['response'] = result['response'] / max_response
            
            # 2. NMS
            response = nms(result['response'], device='cuda')
            
            # 3. Top-K (使用之前定义的 Torch 版本函数)
            # 此时 response 是 (1, 1, H, W)，函数内部应处理 view(-1)
            targets = get_top_k_torch(response, k=1000)
            
        else:
            # --- CPU 处理流程 (保持原有逻辑作为 fallback) ---
            max_response = np.max(result['response'])
            if max_response > 0:
                result['response'] /= max_response
            response = nms(result['response']) # 假设这个 nms 返回 numpy

            response_flat = response.flatten()
            top_n_indices = np.argsort(response_flat)[-1000:][::-1]
            top_n_values = response_flat[top_n_indices]
            top_n_coords = np.unravel_index(top_n_indices, response.shape)
            targets = list(zip(top_n_coords[0], top_n_coords[1], top_n_values))

        # 收集结果 (用于之后的 evaluation)
        video_preds.append(targets)

        # --- 可视化绘制 ---
        # 修正: 只绘制前 show_number 个，防止画面太乱
        for i, target in enumerate(targets):
            if i >= show_number: break 
            
            y, x, score = target
            
            # 转换为整数坐标
            pt = (int(x), int(y))
            cv2.circle(color_img, pt, 5, (255, 0, 0), 2)
            # 画方向线
            # length = 15
            # direction = result['direction'][0, 0, y, x].item() if DEVICE == 'cuda' else result['direction'][y, x]
            # end_pt = (int(x + length * np.cos(direction)), int(y + length * np.sin(direction)))
            # cv2.line(color_img, pt, end_pt, (0, 0, 255), 2)
            # 可选: 加上分数显示
            # cv2.putText(color_img, f"{score:.2f}", (pt[0]+5, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        
        # 添加帧号信息
        cv2.putText(color_img, f'Frame: {frame_idx}', (3300, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        # 显示
        cv2.imshow('Result', color_img)
        
        # 保存
        if is_save and video_writer is not None:
            video_writer.write(color_img) # 修正: 写入视频

        # 键盘控制
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            print("ESC pressed, stopping...")
            break
            


    # 清理资源
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Total Inference Time: {totalTime:.4f}s")
    
    # 返回预测结果供评估使用
    return video_preds


def main(video_id = 1):
    


    sequence_iterator = FrameIterator(os.path.join(ITEM_DIR, 'maritime', 'videos', video_map[f'demo{video_id}']['name']), 
                                      is_video=True)


    # inference_video(sequence_iterator, 
    #                 start_frame=video_map[f'demo{video_id}']['start_frame'],
    #                 end_frame=video_map[f'demo{video_id}']['end_frame'],
    #                 is_save=True,
    #                 show_number=50)
    
    inference_video(sequence_iterator, 
                    start_frame=video_map[f'demo{video_id}']['start_frame'],
                    end_frame=video_map[f'demo{video_id}']['end_frame'],
                    is_save=False,
                    show_number=50)
  

if __name__ == '__main__':
    

    main(video_id=1)