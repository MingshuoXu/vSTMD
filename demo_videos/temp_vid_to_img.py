import os

import cv2
import imageio
import numpy as np

# 使用 moviepy 创建循环 GIF
from moviepy import VideoFileClip

def create_looping_gif(input_path, output_path, loop=0):
    """
    创建循环播放的 GIF
    loop=0: 无限循环
    loop=n: 循环n次
    """
    clip = VideoFileClip(input_path)
    clip.write_gif(output_path, loop=0)  # 0 表示无限循环



def mp4_to_gif_cv2(input_path, output_path, fps=10, max_frames=90):
    """
    使用OpenCV转换MP4到GIF
    """
    # 读取视频
    cap = cv2.VideoCapture(input_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 限制最大帧数
        if max_frames and frame_count >= max_frames:
            break
            
        # 转换BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frames.append(frame_rgb)
        frame_count += 1
    
    cap.release()
    
    # 保存为GIF
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"转换完成！共 {len(frames)} 帧")


# 使用示例
if __name__ == "__main__":
    video_path = "src/vSTMD_F_flying_bird.mp4"  # 输入视频路径
    gif_path = "src/vSTMD_F_flying_bird.gif"         # 输出GIF路径
    # video_path = "src/vSTMD_F_butterfly.mp4"  # 输入视频路径
    # gif_path = "src/vSTMD_F_butterfly.gif"         # 输出GIF路径
    # mp4_to_gif_cv2(video_path, gif_path, fps=10)
    create_looping_gif(video_path, gif_path, loop=0)
