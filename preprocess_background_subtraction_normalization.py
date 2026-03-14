

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 12:35:11 2025

@author: Administrator
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from skimage import exposure
import tifffile
import os

# 更新参数配置
BACKGROUND_PERCENTILE = 70
SPATIAL_FILTER_SIZE = 3
TEMPORAL_WINDOW_SIZE = 21
TEMPORAL_POLY_ORDER = 2

def process_calcium_video(video_path,output_f0,output_df_f0):
    """处理钙成像视频的主函数"""
    
    print("正在读取TIFF视频...")
    # 使用tifffile读取16位TIFF视频
    with tifffile.TiffFile(video_path) as tif:
        video = tif.asarray()
    
    # 确保是3D数组 (帧, 高, 宽)
    if video.ndim == 2:
        video = np.expand_dims(video, axis=0)
    
    print(f"视频加载完成: {video.shape} (帧, 高, 宽), 数据类型: {video.dtype}")
    
    print("计算背景图像...")
    f0 = np.percentile(video, BACKGROUND_PERCENTILE, axis=0)
    
    print("计算 ΔF/F0...")
    # 添加小常数防止除以零
    epsilon = 1e-6 * np.max(video) 
    # epsilon = 1
    df_f0 = (video - f0) / (f0 + epsilon)
     
    print("保存结果...")
    # 保存f0背景图像为16位TIFF
    save_as_16bit_tiff(f0,output_f0)
    
    # 保存df_f0_temporal为16位TIFF视频
    save_as_16bit_tiff(df_f0,output_df_f0)
    
    print("生成对比可视化...")
    visualize_results(video, f0, df_f0,)
    
    return {
        "raw": video,
        "background": f0,
        "df_f0": df_f0
        # "spatial_filtered": df_f0_spatial,
        # "temporal_filtered": df_f0_temporal
    }

def save_as_16bit_tiff(data, filename):
    """将数据保存为16位TIFF格式"""
    # 归一化到0-65535范围
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max - data_min > 1e-6:
        data_norm = (data - data_min) / (data_max - data_min) * 65535
    else:
        data_norm = data * 65535
    
    # 转换为16位无符号整数
    data_16bit = np.clip(data_norm, 0, 65535).astype(np.uint16)
    
    # 保存为TIFF
    tifffile.imwrite(filename, data_16bit)
    print(f"已保存16位TIFF文件: {filename}, 尺寸: {data_16bit.shape}")

def visualize_results(raw, f0, df_f0):
    """创建处理过程的可视化对比"""
    plt.figure(figsize=(18, 12))
    
    # 选择一个代表性的帧
    frame_idx = len(raw) // 2
    
    # 1. 原始帧、背景和ΔF/F0
    plt.subplot(2, 3, 1)
    plt.imshow(raw[frame_idx], cmap='gray')
    plt.title(f"原始帧 #{frame_idx}")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(f0, cmap='gray')
    plt.title("背景估计 (F0)")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(df_f0[frame_idx], cmap='viridis', vmin=-0.5, vmax=2.0)
    plt.title("ΔF/F0")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # 3. 时间序列对比 (选择一个活跃像素)
    active_pixel = find_active_pixel(df_f0)
    t = np.arange(len(raw))
    
    plt.subplot(2, 3, 6)
    plt.plot(t, df_f0[:, active_pixel[0], active_pixel[1]], 'r-', alpha=0.5, label="原始")
    plt.title(f"像素 ({active_pixel[0]}, {active_pixel[1]}) 时间序列")
    plt.xlabel("帧")
    plt.ylabel("ΔF/F0")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("processing_comparison.png", dpi=150)
    plt.show()

def find_active_pixel(df_f0):
    """寻找最活跃的像素"""
    # 计算每个像素的最大ΔF/F0
    max_vals = np.max(df_f0, axis=0)
    # 找到最大值位置
    y, x = np.unravel_index(np.argmax(max_vals), max_vals.shape)
    return (y, x)
 
