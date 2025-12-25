import os

import json
import numpy as np
from moviepy import VideoFileClip, CompositeVideoClip, ColorClip, TextClip
from tqdm import tqdm
import cv2


file_pth = os.path.dirname(os.path.abspath(__file__))



def get_annotation_by_frame_id():
    with open(os.path.join(file_pth, 'annotations', 'instances_train_objects_in_water.json'), 'r') as f:
        data = json.load(f)
        annotation1 = data['annotations']
    with open(os.path.join(file_pth, 'annotations', 'instances_val_objects_in_water.json'), 'r') as f:
        data = json.load(f)
        annotation2 = data['annotations']

    raw_annotations = annotation1 + annotation2
    sort_annotations = sorted(raw_annotations, key=lambda x: x['image_id'])
    annotations_by_frame_id = [[] for _ in range(sort_annotations[-1]['image_id'] + 1)]

    
    for ann in sort_annotations:
        if ann['category_id'] in [1, 2]:  # 只保留 category_id 为 1 和 2 的注释
            frame_id = ann['image_id']
            annotations_by_frame_id[frame_id].append(ann)

    return annotations_by_frame_id



def create_frame_by_frame_comparison(
    orig_path, 
    ann_path, 
    model_path, 
    output_path, 
    roi_list,      
    start_frame,
    zoom_left=True,
    zoom_factor=2.0,
    border_width=10
):
    print(f"正在处理: {orig_path}")
    
    # 1. 加载视频
    clip_orig = VideoFileClip(orig_path)
    clip_ann = VideoFileClip(ann_path)
    clip_model = VideoFileClip(model_path)
    
    # 获取 FPS (用于将时间 t 转换为 列表索引)
    fps = clip_orig.fps
    if fps is None:
        fps = 30 # 默认兜底
        print("警告: 无法读取 FPS，默认使用 30")

    # 2. 确保视频长度与 ROI 列表匹配
    # 计算 ROI 列表能覆盖的时长
    roi_duration = len(roi_list) / fps
    
    # 取最小公共时长
    min_duration = min(clip_orig.duration, clip_ann.duration, clip_model.duration, roi_duration)
    
    # 截取
    clip_orig = clip_orig.subclipped(0, min_duration)
    clip_ann = clip_ann.subclipped(0, min_duration)
    clip_model = clip_model.subclipped(0, min_duration)

    print(f"视频 FPS: {fps}, 总帧数: {len(roi_list)}, 截取时长: {min_duration:.2f}s")

    # 3. 辅助函数：逐帧动态裁剪
    def process_roi(clip, label_text, color):
        
        # 定义核心 Filter 函数
        def crop_filter(get_frame, t):
            # 获取当前画面
            frame = get_frame(t) 
            
            # --- 核心逻辑：将时间 t 映射到 roi_list 的索引 ---
            frame_idx = int(t * fps)
            
            # 索引保护：防止超出列表范围
            if frame_idx >= len(roi_list):
                frame_idx = len(roi_list) - 1
            
            # 获取当前帧对应的坐标
            x, y, w, h = roi_list[frame_idx]
            
            # 坐标整数化
            x, y, w, h = int(x), int(y), int(w), int(h)

            # 边界保护 (防止 ROI 超出图像边缘报错)
            img_h, img_w = frame.shape[:2]
            y1 = max(0, min(y, img_h - 1))
            y2 = max(0, min(y + h, img_h))
            x1 = max(0, min(x, img_w - 1))
            x2 = max(0, min(x + w, img_w))
            
            # 裁剪
            cropped_img = frame[y1:y2, x1:x2]
            
            # 尺寸修正：如果切出来的图比 w, h 小（在边缘时），补黑边
            # 这一步非常重要，否则 resize 会因为每一帧尺寸不同而报错或抖动
            if cropped_img.shape[0] != h or cropped_img.shape[1] != w:
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                # 能够切出来的有效区域大小
                valid_h, valid_w = cropped_img.shape[:2]
                padded[:valid_h, :valid_w] = cropped_img
                return padded
                
            return cropped_img

        # 应用动态裁剪
        cropped_clip = clip.transform(crop_filter).with_duration(clip.duration)
        

        target_w = roi_list[0][2] * zoom_factor
        zoomed = cropped_clip.resized(width=target_w)
        
        # 制作边框
        border = ColorClip(
            size=(int(zoomed.w + 2*border_width), int(zoomed.h + 2*border_width)), 
            color=color,
            duration=clip.duration
        )
        
        zoomed = zoomed.with_position((border_width, border_width))
        

        txt = TextClip(
            text=label_text, 
            font_size=56, 
            color=color, 
            method='caption',
            size=(int(target_w), 80),
            duration=clip.duration
        ).with_position(("center", 0))
         
        final_piece = CompositeVideoClip([border, zoomed, txt])
            
        return final_piece

    # 4. 生成三个组件
    roi_orig = process_roi(clip_orig, "Original", color=(0, 0, 0))
    roi_model = process_roi(clip_model, "Model Output", color=(0, 0, 255))
    roi_ann = process_roi(clip_ann, "Annotation", color=(0, 255, 0))

    # 5. 布局位置
    margin = 10
    total_h = roi_orig.h
    
    if zoom_left:
        roi_orig = roi_orig.with_position((margin+20, margin+80))
        roi_model = roi_model.with_position((margin+20, margin + total_h + margin+80))
        roi_ann = roi_ann.with_position((margin+20, margin + (total_h + margin) * 2+80))
    else:
        zoomed_in_w = roi_list[0][2] * zoom_factor + 2*border_width
        roi_orig = roi_orig.with_position((3840-zoomed_in_w-margin, margin+80))
        roi_model = roi_model.with_position((3840-zoomed_in_w-margin, margin + total_h + margin+80))
        roi_ann = roi_ann.with_position((3840-zoomed_in_w-margin, margin + (total_h + margin) * 2+80))

    # 6. 合成并导出
    final_video = CompositeVideoClip([
        clip_orig,
        roi_orig,
        roi_model,
        roi_ann
    ])

    def add_frame_counter(get_frame, t, start_frame=start_frame):
        """
        动态绘制帧编号的 Filter 函数
        get_frame: 获取当前时间 t 的图像函数
        t: 当前时间 (秒)
        """
        # 1. 获取当前帧图像 (numpy array)
        frame = get_frame(t)
        # 确保图像是可写的副本
        frame = np.array(frame)
        
        # 2. 计算当前帧号
        current_frame = int(t * fps) + start_frame
        text = f"Frame: {current_frame}"
        
        # 3. 设置字体参数 (OpenCV)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5      # 字体大小
        thickness = 2         # 字体粗细
        color_fg = (255, 255, 255) # 白色字
        color_bg = (0, 0, 0)       # 黑色描边
        
        # 4. 计算文字尺寸以居中
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_w, text_h = text_size
        img_h, img_w = frame.shape[:2]
        
        x = (img_w - text_w) // 2  # 水平居中
        y = 50                     # 距离顶部的距离 (像素)

        # 5. 绘制文字 (先画黑色描边，再画白色文字，确保任何背景下可见)
        cv2.putText(frame, text, (x, y), font, font_scale, color_bg, thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, font_scale, color_fg, thickness, cv2.LINE_AA)
        
        return frame
    
    final_video = final_video.transform(add_frame_counter)

    print(f"正在导出到 {output_path} ...")
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    clip_orig.close()
    clip_ann.close()
    clip_model.close()


def get_zoomed_roi(start_frame, end_frame, width=700, height=260, top_fix = 50, left_fix = 50):
    anno = get_annotation_by_frame_id()

    zoomed_roi = [None for _ in range(end_frame - start_frame)]
    for ann in anno[start_frame:end_frame]:
        top = 2160
        left = 3840
        for a in ann:
            if a['category_id'] in [1, 2]:
                x, y, w, h = a['bbox']
                top = min(top, y)
                left = min(left, x)
        top = min(2160-height, top-top_fix)
        left = min(3840-width, left-left_fix)
        zoomed_roi[ann[0]['image_id'] - start_frame] = (left, top, width, height)


    return zoomed_roi


if __name__ == "__main__":
    # 配置参数

    
    file_pth = os.path.dirname(os.path.abspath(__file__))

    video_map = {
        'demo1': {'name': 'demo1-SeaDronesSee-696-1410.mp4', 
                  'start_frame': 696, 'end_frame': 1410,
                  'zoomed_left': True,
                  'zoomed_top_fix': 50, 'zoomed_left_fix': 0,
                  'zoomed_width': 800, 'zoomed_height': 300,
                  'zoomed_in_factor': 2},
        'demo2': {'name': 'demo2-SeaDronesSee-1697-2411.mp4',
                    'start_frame': 1697, 'end_frame': 2411,
                    'zoomed_left': True,
                    'zoomed_top_fix': 100, 'zoomed_left_fix': 150,
                    'zoomed_width': 300, 'zoomed_height': 200,
                  'zoomed_in_factor': 2.5},
        'demo3': {'name': 'demo3-SeaDronesSee-3666-4166.mp4',
                    'start_frame': 3666, 'end_frame': 4166,
                    'zoomed_left': False,
                    'zoomed_top_fix': 100, 'zoomed_left_fix': 150,
                    'zoomed_width': 500, 'zoomed_height': 260,
                  'zoomed_in_factor': 2.5},
        'demo4': {'name': 'demo4-SeaDronesSee-22931-23545.mp4',
                    'start_frame': 22931, 'end_frame': 23545,
                    'zoomed_left': False,
                    'zoomed_top_fix': 150, 'zoomed_left_fix': 100,
                    'zoomed_width': 700, 'zoomed_height': 300,
                  'zoomed_in_factor': 2},
        'demo5': {'name': 'demo5-SeaDronesSee-29713-30312.mp4',
                    'start_frame': 29713, 'end_frame': 30312,
                    'zoomed_left': True,
                    'zoomed_top_fix': 50, 'zoomed_left_fix': 50,
                    'zoomed_width': 700, 'zoomed_height': 400,
                  'zoomed_in_factor': 1.5},
    }

    for key, val in tqdm(video_map.items()):
        zoomed_roi = get_zoomed_roi(val['start_frame'], val['end_frame'],
                                    width=val['zoomed_width'], height=val['zoomed_height'],
                                    top_fix=val.get('zoomed_top_fix', 0))

        create_frame_by_frame_comparison(
            orig_path=os.path.join(file_pth, 'videos', 
                                   f"{key}-SeaDronesSee-{val['start_frame']}-{val['end_frame']}.mp4"),
            ann_path=os.path.join(file_pth, 'results', f"{val['start_frame']}_anno.mp4"),
            model_path=os.path.join(file_pth, 'results', f"inference_output_{val['start_frame']}_{val['end_frame']}.mp4"),
            output_path=os.path.join(file_pth, 'results', f"reselt_{val['start_frame']}_{val['end_frame']}.mp4"),
            start_frame=val['start_frame'],
            zoom_left=val['zoomed_left'],
            roi_list=zoomed_roi,
            zoom_factor=val['zoomed_in_factor']
        )

  