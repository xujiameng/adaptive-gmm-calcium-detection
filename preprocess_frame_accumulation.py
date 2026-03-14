# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 20:47:07 2025

@author: Administrator
"""

import cv2
import numpy as np
import os
from collections import deque
import time
import threading
from tqdm import tqdm


 

def sliding_window_average(input_path, output_path, window_size=5, step=1, 
                          save_quality='lossless', progress_callback=None):
    """
    对AVI视频进行滑动平均处理，支持大文件处理和高保真保存
    
    参数:
        input_path: 输入AVI文件路径
        output_path: 输出文件路径
        window_size: 滑动窗口大小 (默认5)
        step: 滑动步长 (默认1)
        save_quality: 保存质量选项 ('high', 'lossless') (默认'lossless')
        progress_callback: 进度回调函数 (function(current_frame, total_frames))
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {input_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
     
    
    # 检查位深度并确保高质量处理
    bit_depth = int(cap.get(cv2.CAP_PROP_FORMAT))
    if bit_depth < 0:  # 如果无法获取位深度，默认为8位
        bit_depth = 8
    
    # 计算输出帧率
    output_fps = original_fps / step if step > 1 else original_fps
    
    # 创建高质量视频写入器
    fourcc, extension = get_quality_settings(save_quality, output_path, bit_depth)
    
    # 确保输出路径使用正确的扩展名
    if not output_path.endswith(extension):
        output_path = os.path.splitext(output_path)[0] + extension
    
    # 创建视频写入器
    if bit_depth > 8:
        # 对于高位深视频，使用无损格式
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height), isColor=True)
        # 使用32位浮点进行累加以保持精度
        dtype = np.float32
    else:
        # 对于8位视频，使用高质量编码
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height), isColor=True)
        dtype = np.float32  # 仍然使用浮点累加以保持精度
    
    if not out.isOpened():
        cap.release()
        raise IOError(f"无法创建输出文件: {output_path} (尝试使用fourcc: {fourcc})")
    
    # 设置视频质量参数（如果支持）
    if save_quality == 'high':
        # 设置高质量编码参数
        out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)  # 最高质量
    elif save_quality == 'lossless' and bit_depth <= 8:
        # 对于8位无损，设置无损参数
        out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    
    print(f"视频处理设置: 输入位深={bit_depth}位, 保存质量={save_quality}, 编码器={fourcc_to_name(fourcc)}")
    
    # 初始化滑动窗口 
    
    frame_buffer = deque(maxlen=window_size)
    frame_sum = None
    processed_frames = 0
    pbar = tqdm(total=total_frames, desc="处理视频帧", unit="帧")
    
    # 处理第一窗口
    for _ in range(window_size):
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
        
        # 初始化累加器
        if frame_sum is None:
            frame_sum = np.zeros_like(frame, dtype=dtype)
        frame_sum += frame.astype(dtype)
        
        # 更新进度条 
        pbar.update(1)
    
    # 如果成功读取完整窗口，写入第一帧
    if len(frame_buffer) == window_size:
        avg_frame = (frame_sum / window_size).astype(np.uint16 if bit_depth > 8 else np.uint8)
        out.write(avg_frame)
        processed_frames += 1
        if progress_callback:
            progress_callback(processed_frames, total_frames)
    
    # 主处理循环
    skip_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 更新进度条 
        pbar.update(1)
        
        # 更新滑动窗口
        oldest_frame = frame_buffer.popleft()
        frame_buffer.append(frame)
        
        # 更新累加器
        frame_sum -= oldest_frame.astype(dtype)
        frame_sum += frame.astype(dtype)
        
        skip_counter += 1
        
        # 根据步长决定是否写入
        if skip_counter >= step:
            avg_frame = (frame_sum / window_size).astype(np.uint16 if bit_depth > 8 else np.uint8)
            out.write(avg_frame)
            processed_frames += 1
            skip_counter = 0
            
            if progress_callback and processed_frames % 10 == 0:
                progress_callback(processed_frames, total_frames)
    
    # 处理剩余帧
    while len(frame_buffer) > 1 and skip_counter > 0:
        # 更新进度条（剩余帧） 
        pbar.update(1)
        
        oldest_frame = frame_buffer.popleft()
        frame_sum -= oldest_frame.astype(dtype)
        
        avg_frame = (frame_sum / len(frame_buffer)).astype(np.uint16 if bit_depth > 8 else np.uint8)
        out.write(avg_frame)
        processed_frames += 1
        
        if progress_callback:
            progress_callback(processed_frames, total_frames)
    
    # 关闭进度条 
    pbar.close()
    
    # 释放资源
    cap.release()
    out.release()
    
    return {
        'original_frames': total_frames,
        'processed_frames': processed_frames,
        'output_fps': output_fps,
        'window_size': window_size,
        'step': step,
        'bit_depth': bit_depth,
        'output_path': output_path,
        'codec': fourcc_to_name(fourcc),
        'quality': save_quality
    }

 


def play_video(video_path, fps=None, window_name="Processed Video"):
    """
    播放处理后的视频，可调整播放速度
    
    参数:
        video_path: 视频文件路径
        fps: 自定义帧率 (None表示使用原始帧率)
        window_name: 播放窗口名称
    """
    # 检查是否为TIFF序列
    if video_path.lower().endswith('.tif') or video_path.lower().endswith('.tiff'):
        return play_tiff_sequence(video_path, fps, window_name)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 设置播放速度
    playback_fps = fps if fps is not None else original_fps
    frame_delay = max(1, int(1000 / playback_fps))  # 毫秒
    
    current_frame = 0
    paused = False
    
    print(f"播放控制: [Space]暂停/继续 | [Right]下一帧 | [Left]上一帧 | [+]加速 [-]减速 | [Esc]退出")
    print(f"视频信息: {width}x{height} @ {original_fps:.1f}fps, 总帧数: {total_frames}")
    
    while True:
        start_time = time.time()
        
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # 循环播放
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame = 0
                continue
            
            current_frame += 1
        else:
            # 暂停时保持当前帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
        
        if ret:
            # 显示帧信息和进度条
            info_text = f"Frame: {current_frame}/{total_frames} | FPS: {playback_fps:.1f}"
            cv2.putText(frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制进度条
            progress = current_frame / total_frames
            cv2.rectangle(frame, (0, height-10), 
                          (int(width * progress), height), 
                          (0, 255, 0), -1)
            
            cv2.imshow(window_name, frame)
        
        # 计算实际处理时间并调整延迟
        processing_time = (time.time() - start_time) * 1000  # 毫秒
        adjusted_delay = max(1, int(frame_delay - processing_time))
        
        # 键盘控制
        key = cv2.waitKey(adjusted_delay) & 0xFF
        
        if key == 27:  # ESC键
            break
        elif key == 32:  # 空格键
            paused = not paused
        elif key == 83:  # 右箭头
            current_frame = min(current_frame + 1, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        elif key == 81:  # 左箭头
            current_frame = max(0, current_frame - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        elif key == ord('+') or key == ord('='):  # 加速
            playback_fps = min(120, playback_fps + 5)
            frame_delay = max(1, int(1000 / playback_fps))
            print(f"播放速度: {playback_fps:.1f}fps")
        elif key == ord('-') or key == ord('_'):  # 减速
            playback_fps = max(0.5, playback_fps - 5)
            frame_delay = max(1, int(1000 / playback_fps))
            print(f"播放速度: {playback_fps:.1f}fps")
    
    cap.release()
    cv2.destroyAllWindows()


def play_tiff_sequence(folder_path, fps=30, window_name="TIFF Sequence"):
    """
    播放TIFF序列（用于高位深无损保存的视频）
    
    参数:
        folder_path: 包含TIFF文件的文件夹路径
        fps: 播放帧率
        window_name: 播放窗口名称
    """
    # 获取所有TIFF文件
    tiff_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.tif', '.tiff'))]
    tiff_files.sort()
    
    if not tiff_files:
        raise FileNotFoundError(f"未找到TIFF文件: {folder_path}")
    
    total_frames = len(tiff_files)
    current_frame = 0
    paused = False
    frame_delay = max(1, int(1000 / fps))
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(os.path.join(folder_path, tiff_files[0]), cv2.IMREAD_ANYDEPTH)
    height, width = first_frame.shape[:2]
    
    print(f"播放TIFF序列: {total_frames}帧 @ {width}x{height}")
    print(f"播放控制: [Space]暂停/继续 | [Right]下一帧 | [Left]上一帧 | [+]加速 [-]减速 | [Esc]退出")
    
    while True:
        start_time = time.time()
        
        if not paused:
            frame_path = os.path.join(folder_path, tiff_files[current_frame])
            frame = cv2.imread(frame_path, cv2.IMREAD_ANYDEPTH)
            
            # 转换为8位用于显示（保留原始数据）
            display_frame = convert_to_8bit(frame)
            
            # 显示帧信息
            info_text = f"Frame: {current_frame+1}/{total_frames} | FPS: {fps:.1f}"
            cv2.putText(display_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制进度条
            progress = (current_frame + 1) / total_frames
            cv2.rectangle(display_frame, (0, height-10), 
                          (int(width * progress), height), 
                          (0, 255, 0), -1)
            
            cv2.imshow(window_name, display_frame)
            
            current_frame = (current_frame + 1) % total_frames
        
        # 键盘控制
        key = cv2.waitKey(frame_delay) & 0xFF
        
        if key == 27:  # ESC键
            break
        elif key == 32:  # 空格键
            paused = not paused
        elif key == 83:  # 右箭头
            current_frame = min(current_frame + 1, total_frames - 1)
        elif key == 81:  # 左箭头
            current_frame = max(0, current_frame - 1)
        elif key == ord('+') or key == ord('='):  # 加速
            fps = min(120, fps + 5)
            frame_delay = max(1, int(1000 / fps))
            print(f"播放速度: {fps:.1f}fps")
        elif key == ord('-') or key == ord('_'):  # 减速
            fps = max(0.5, fps - 5)
            frame_delay = max(1, int(1000 / fps))
            print(f"播放速度: {fps:.1f}fps")
    
    cv2.destroyAllWindows()


def convert_to_8bit(image):
    """将高位深图像转换为8位用于显示，保持最大动态范围"""
    if image.dtype == np.uint8:
        return image
    
    # 自动调整对比度
    min_val = np.min(image)
    max_val = np.max(image)
    
    if min_val == max_val:
        return np.zeros_like(image, dtype=np.uint8)
    
    # 线性拉伸到0-255
    return (((image - min_val) / (max_val - min_val)) * 255).astype(np.uint8)



 

def get_quality_settings(quality, output_path, bit_depth):
    """根据质量要求和位深度返回合适的编码器和文件扩展名"""
    base_name, ext = os.path.splitext(output_path)
    ext = ext.lower()
    
    if quality == 'lossless':
        # 无损保存选项 - 使用更广泛支持的编码
        if bit_depth > 8:
            # 高位深视频使用TIFF序列
            return 0, '.tif'  # 特殊处理
        else:
            # 8位视频使用Motion JPEG无损编码 (Fiji兼容)
            return cv2.VideoWriter_fourcc(*'MJPG'), '.avi'
    
    # 高质量保存选项
    if ext in ['.mp4', '.m4v']:
        # 使用H.265 (HEVC) 高质量编码
        return cv2.VideoWriter_fourcc(*'HEVC'), '.mp4'
    elif ext in ['.avi']:
        # AVI容器使用Motion JPEG编码 (Fiji兼容)
        return cv2.VideoWriter_fourcc(*'MJPG'), '.avi'
    else:
        # 默认使用H.264编码
        return cv2.VideoWriter_fourcc(*'H264'), '.mp4'


def fourcc_to_name(fourcc):
    """将fourcc代码转换为可读名称"""
    if fourcc == cv2.VideoWriter_fourcc(*'MJPG'):
        return "Motion JPEG (Lossless)"
    elif fourcc == cv2.VideoWriter_fourcc(*'H264'):
        return "H.264/AVC"
    elif fourcc == cv2.VideoWriter_fourcc(*'HEVC'):
        return "H.265/HEVC"
    elif fourcc == cv2.VideoWriter_fourcc(*'ap4h'):
        return "Apple ProRes 422 HQ"
    else:
        return f"Unknown (code: {fourcc})"








 
# import cv2
# import numpy as np
# import os
# from collections import deque
# import time
import glob
import tifffile
# from tqdm import tqdm

def frame_accumulation(input_path, output_path, window_size=8, step=1, 
                      progress_callback=None):
    """
    对视频/图像序列进行帧累加处理，保存为多页16位TIFF视频
    
    参数:
        input_path: 输入路径（视频文件或包含图像的文件夹）
        output_path: 输出TIFF文件路径
        window_size: 滑动窗口大小 (默认8)
        step: 滑动步长 (默认1)
        progress_callback: 进度回调函数 (function(current_frame, total_frames))
    """
    # 确保输出路径是TIFF格式
    if not output_path.lower().endswith(('.tif', '.tiff')):
        output_path = os.path.splitext(output_path)[0] + '.tif'
    
    # 获取帧序列（根据输入类型）
    if os.path.isfile(input_path):
        # 视频文件
        if input_path.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
            frames = read_video_frames(input_path)
        elif input_path.lower().endswith(('.tif', '.tiff')):
            frames = read_tiff_sequence(input_path)
        else:
            raise ValueError(f"不支持的视频格式: {input_path}")
    elif os.path.isdir(input_path):
        # 图像序列文件夹
        frames = read_image_sequence(input_path)
    else:
        raise FileNotFoundError(f"路径不存在: {input_path}")
    
    # 获取总帧数
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError("未找到可处理的帧")
    
    print(f"开始帧累加处理: 总帧数={total_frames}, 窗口大小={window_size}, 步长={step}")
    
    # 初始化滑动窗口和累加器
    frame_buffer = deque(maxlen=window_size)
    accumulator = None
    processed_frames = 0
    
    # 创建进度条
    pbar = tqdm(total=total_frames, desc="帧累加处理", unit="帧")
    
    # 准备输出TIFF文件
    output_frames = []
    
    # 处理每一帧
    for i, frame in enumerate(frames):
        # 预处理帧（转换为8位灰度）
        processed_frame = preprocess_frame(frame)
        
        # 转换为16位进行累加
        frame_16bit = processed_frame.astype(np.uint16)
        
        # 更新累加器
        if accumulator is None:
            # 初始化累加器
            accumulator = np.zeros_like(frame_16bit, dtype=np.uint32)
            height, width = processed_frame.shape
            print(f"图像尺寸: {width}x{height}")
        
        # 更新滑动窗口
        if len(frame_buffer) == window_size:
            # 移除最旧帧
            oldest_frame = frame_buffer.popleft()
            accumulator -= oldest_frame
        
        # 添加新帧
        frame_buffer.append(frame_16bit)
        accumulator += frame_16bit
        
        # 更新进度条
        pbar.update(1)
        
        # 当窗口满时输出结果
        if len(frame_buffer) == window_size and (i % step == 0 or step == 1):
            # 计算平均值
            avg_frame = (accumulator / window_size).astype(np.uint16)
            
            # 添加到输出帧列表
            output_frames.append(avg_frame)
            processed_frames += 1
            
            if progress_callback:
                progress_callback(processed_frames, total_frames)
    
    # 关闭进度条
    pbar.close()
    
    # 保存为多页TIFF
    if output_frames:
        print(f"正在保存为多页TIFF: {output_path} (包含{len(output_frames)}帧)")
        
        # 转换为3D数组 (frames, height, width)
        tiff_stack = np.stack(output_frames, axis=0)
        
        # 保存为多页TIFF
        tifffile.imwrite(
            output_path,
            tiff_stack,
            photometric='minisblack',  # 灰度图像
            metadata={'axes': 'TYX'}   # 时间序列，然后是Y和X维度
        )
        
        print(f"处理完成: 输出{len(output_frames)}帧到{output_path}")
    else:
        print("警告: 未生成任何帧，请检查窗口大小和步长设置")
    
    return {
        'original_frames': total_frames,
        'processed_frames': len(output_frames),
        'window_size': window_size,
        'step': step,
        'output_path': output_path,
        'dimensions': f"{height}x{width}"
    }


def preprocess_frame(frame):
    """预处理帧：转换为8位灰度图像"""
    # 确保是2D数组（灰度图像）
    if len(frame.shape) == 3:
        # 如果是彩色图像，转换为灰度
        if frame.shape[2] == 3:  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        else:
            # 其他通道数，取第一个通道
            frame = frame[:, :, 0]
    
    # 确保是8位图像
    if frame.dtype != np.uint8:
        # 处理16位图像：线性缩放到8位
        if frame.dtype == np.uint16:
            frame = (frame / 256).astype(np.uint8)
        # 处理浮点图像：缩放到0-255
        elif frame.dtype == np.float32 or frame.dtype == np.float64:
            min_val = np.min(frame)
            max_val = np.max(frame)
            if max_val > min_val:
                frame = ((frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    
    return frame


def read_video_frames(video_path):
    """从视频文件中读取所有帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height} @ {fps:.1f}fps, 总帧数: {total_frames}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 将BGR转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    print(f"从视频中读取 {len(frames)} 帧")
    return frames


def read_tiff_sequence(tiff_path):
    """读取TIFF文件（单文件多页或多文件序列）"""
    if os.path.isfile(tiff_path):
        # 单文件多页TIFF
        print(f"读取多页TIFF文件: {tiff_path}")
        with tifffile.TiffFile(tiff_path) as tif:
            frames = tif.asarray()
            
            # 检查TIFF维度
            if frames.ndim == 2:
                # 单帧
                frames = [frames]
            elif frames.ndim == 3:
                # 多帧灰度
                frames = [frames[i] for i in range(frames.shape[0])]
            elif frames.ndim == 4:
                # 多帧彩色 - 转换为灰度
                frames = [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(frames.shape[0])]
            
        print(f"从TIFF文件中读取 {len(frames)} 帧")
        return frames
    else:
        raise FileNotFoundError(f"TIFF文件不存在: {tiff_path}")


def read_image_sequence(folder_path):
    """从文件夹中读取图像序列（支持多种格式）"""
    # 支持的文件格式
    extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.jp2']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    image_files.sort()
    
    if not image_files:
        raise FileNotFoundError(f"未找到图像文件: {folder_path}")
    
    frames = []
    for file_path in image_files:
        # 使用OpenCV读取图像
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # 如果OpenCV无法读取，尝试用tifffile
            if file_path.lower().endswith(('.tif', '.tiff')):
                try:
                    img = tifffile.imread(file_path)
                except:
                    print(f"无法读取图像: {file_path}")
                    continue
            else:
                print(f"无法读取图像: {file_path}")
                continue
        
        # 将BGR转换为RGB（如果是彩色）
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    
    print(f"从文件夹中读取 {len(frames)} 帧")
    return frames


def play_tiff_video(tiff_path, fps=30, window_name="TIFF Video"):
    """
    播放多页TIFF视频
    
    参数:
        tiff_path: TIFF文件路径
        fps: 播放帧率
        window_name: 播放窗口名称
    """
    # 读取TIFF文件
    print(f"加载TIFF视频: {tiff_path}")
    with tifffile.TiffFile(tiff_path) as tif:
        video_frames = tif.asarray()
    
    # 检查TIFF维度
    if video_frames.ndim == 2:
        # 单帧
        video_frames = [video_frames]
    elif video_frames.ndim == 3:
        # 多帧灰度
        video_frames = [video_frames[i] for i in range(video_frames.shape[0])]
    elif video_frames.ndim == 4:
        # 多帧彩色 - 转换为灰度
        video_frames = [cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2GRAY) for i in range(video_frames.shape[0])]
    
    total_frames = len(video_frames)
    height, width = video_frames[0].shape[:2]
    
    print(f"视频信息: {width}x{height}, 总帧数: {total_frames}")
    
    current_frame = 0
    paused = False
    frame_delay = max(1, int(1000 / fps))
    
    print(f"播放控制: [Space]暂停/继续 | [Right]下一帧 | [Left]上一帧 | [+]加速 [-]减速 | [Esc]退出")
    
    while True:
        start_time = time.time()
        
        # 获取当前帧
        frame = video_frames[current_frame]
        
        # 转换为8位用于显示
        display_frame = convert_to_8bit(frame)
        
        # 显示帧信息
        info_text = f"Frame: {current_frame+1}/{total_frames} | FPS: {fps:.1f}"
        cv2.putText(display_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制进度条
        progress = (current_frame + 1) / total_frames
        cv2.rectangle(display_frame, (0, height-10), 
                      (int(width * progress), height), 
                      (0, 255, 0), -1)
        
        cv2.imshow(window_name, display_frame)
        
        # 键盘控制
        key = cv2.waitKey(frame_delay) & 0xFF
        
        if key == 27:  # ESC键
            break
        elif key == 32:  # 空格键
            paused = not paused
        elif key == 83 or key == 3:  # 右箭头或D键
            current_frame = min(current_frame + 1, total_frames - 1)
        elif key == 81 or key == 2:  # 左箭头或A键
            current_frame = max(0, current_frame - 1)
        elif key == ord('+') or key == ord('='):  # 加速
            fps = min(120, fps + 5)
            frame_delay = max(1, int(1000 / fps))
            print(f"播放速度: {fps:.1f}fps")
        elif key == ord('-') or key == ord('_'):  # 减速
            fps = max(0.5, fps - 5)
            frame_delay = max(1, int(1000 / fps))
            print(f"播放速度: {fps:.1f}fps")
        
        # 如果不是暂停状态，前进到下一帧
        if not paused:
            current_frame = (current_frame + 1) % total_frames
    
    cv2.destroyAllWindows()


def convert_to_8bit(image):
    """将高位深图像转换为8位用于显示，保持最大动态范围"""
    if image.dtype == np.uint8:
        return image
    
    # 自动调整对比度
    min_val = np.min(image)
    max_val = np.max(image)
    
    if min_val == max_val:
        return np.zeros_like(image, dtype=np.uint8)
    
    # 线性拉伸到0-255
    return (((image - min_val) / (max_val - min_val)) * 255).astype(np.uint8)

 
 
import tifffile as tif
import sys 

def output_comparison_video(WIN,STEP,in_path,out_path): 
    stack = tif.imread(in_path)          # TxHxW
    T, H, W = stack.shape

    # 1. 计算累加序列的合法帧数
    k_max = (T - WIN) // STEP + 1
    centers = np.arange(k_max) * STEP + WIN // 2   # 中心索引
    aligned = stack[centers]                       # 与累加结果一一对应

    tif.imwrite(out_path, aligned.astype(stack.dtype))
    print(f"Aligned original: {aligned.shape[0]} frames -> {out_path}")



