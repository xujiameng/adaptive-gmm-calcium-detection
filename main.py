# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 12:20:29 2025

@author: Administrator
"""
   
    
#%% # # # # # # # # # # # # # # # # # # # # # # # # # # #  预处理——帧累加  # # # # # # # # # # # # # # # # # # # # # # # # # # #  
# # 帧累加 
window_size = 13  

from 预处理_帧累加 import sliding_window_average,play_video,frame_accumulation
 
result = frame_accumulation(
    "含噪声影像_演示示例.tif",
    "./钙事件分割过程结果/含噪声影像_演示示例_预处理_帧累加.tif",
    window_size=window_size,
    step=1  
)

 

#%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  预处理——背景减除与归一化 # # # # # # # # # # # # # # # # # # # # # # # # # # #  
  
## ## 计算一张背景静态f0，并计算（f-f0）/f0,获得df/f0
from 预处理_背景减除与归一化 import process_calcium_video
 
video_path = './钙事件分割过程结果/含噪声影像_演示示例_预处理_帧累加.tif'
output_f0 = './钙事件分割过程结果/含噪声影像_演示示例_f0.tif'
output_df_f0 = './钙事件分割过程结果/含噪声影像_演示示例_预处理_背景剔除与归一化.tif'
results = process_calcium_video(video_path,output_f0,output_df_f0)
print("处理完成!")
  

#%% # # # # # # # # # # # # # # # # # # # # # # # # # # #  预处理——高斯滤波  # # # # # # # # # # # # # # # # # # # # # # # # # # #  
import numpy as np
from skimage.filters import median
from skimage.morphology import disk
import tifffile as tif
from pathlib import Path

def median_filter_tif_video(
        in_path: str,
        out_path: str,
        radius: int = 1,
        selem: np.ndarray = None,
        verbose: bool = True
    ):
    in_path, out_path = Path(in_path), Path(out_path)

    # 1. 读取源文件
    with tif.TiffFile(in_path) as src:
        frames = src.asarray()
        dtype  = frames.dtype
        # ------------ 兼容旧版 tifffile ------------
        # shaped_metadata() 不存在时，用 pages[0].tags 构造一个简单 dict
        try:
            meta = src.shaped_metadata()        # 新版
        except (TypeError, AttributeError):
            # 旧版：把 ImageDescription 等常用字段取出来即可
            meta = {tag.name: tag.value
                    for tag in src.pages[0].tags.values()
                    if tag.name in ('ImageDescription', 'Software', 'DateTime')}
        # ------------------------------------------
        if verbose:
            print(f"[info] 原始视频 shape: {frames.shape}, dtype: {dtype}")

    # 2. 构造结构元
    if selem is None:
        selem = disk(radius)

    # 3. 逐帧中值滤波
    if verbose:
        print("[info] 开始中值滤波 ...")
    filtered = np.empty_like(frames, dtype=dtype)

    if frames.ndim == 3:                 # (T, H, W)
        for t, img in enumerate(frames):
            if verbose and t % 100 == 0:
                print(f"    processed frame {t}/{frames.shape[0]}")
            filtered[t] = median(img, footprint=selem)
    elif frames.ndim == 4 and frames.shape[-1] == 1:   # (T, H, W, 1)
        for t, img in enumerate(frames):
            if verbose and t % 100 == 0:
                print(f"    processed frame {t}/{frames.shape[0]}")
            filtered[t, ..., 0] = median(img[..., 0], footprint=selem)
    else:
        raise ValueError("仅支持灰度或单通道 tif 视频 (T, H, W) 或 (T, H, W, 1)")

    # 4. 保存
    tif.imwrite(
        out_path,
        filtered,
        bigtiff=False,
        metadata=meta if meta else None,
        imagej=getattr(src, 'is_imagej', None)
    )
    if verbose:
        print(f"[info] 已保存 -> {out_path}")



import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import tifffile as tif

def gaussian_filter_tif_video(
        in_path: str,
        out_path: str,
        sigma: float = 1.0,
        mode: str = 'reflect',
        truncate: float = 4.0,
        verbose: bool = True
    ):
    """
    逐帧高斯滤波，并保持位深、dtype、元数据不变地保存为新的 tif 视频。

    Parameters
    ----------
    in_path : str
        输入 tif 视频路径
    out_path : str
        输出 tif 视频路径
    sigma : float or sequence of float
        高斯核标准差（像素单位）
    mode : str
        边界填充方式，同 scipy.ndimage.gaussian_filter
    truncate : float
        截断半径，默认 4σ
    verbose : bool
        是否打印进度

    Returns
    -------
    None
    """
    in_path, out_path = Path(in_path), Path(out_path)

    # 1. 读取
    with tif.TiffFile(in_path) as src:
        frames = src.asarray()        # (T, H, W) 或 (T, H, W, C)
        dtype  = frames.dtype
        meta   = {tag.name: tag.value
                  for tag in src.pages[0].tags.values()} if src.pages else {}
        imagej = src.is_imagej if hasattr(src, 'is_imagej') else None
        if verbose:
            print(f"[info] 原始视频 shape: {frames.shape}, dtype: {dtype}")

    # 2. 逐帧滤波
    if verbose:
        print("[info] 开始高斯滤波 ...")
    filtered = np.empty_like(frames, dtype=dtype)

    # 支持灰度或单通道
    if frames.ndim == 3:                 # (T, H, W)
        for t, img in enumerate(frames):
            if verbose and t % 100 == 0:
                print(f"    processed frame {t}/{frames.shape[0]}")
            filtered[t] = gaussian_filter(img, sigma=sigma,
                                          mode=mode, truncate=truncate)
    elif frames.ndim == 4 and frames.shape[-1] == 1:   # (T, H, W, 1)
        for t, img in enumerate(frames):
            if verbose and t % 100 == 0:
                print(f"    processed frame {t}/{frames.shape[0]}")
            filtered[t, ..., 0] = gaussian_filter(img[..., 0],
                                                  sigma=sigma,
                                                  mode=mode,
                                                  truncate=truncate)
    else:
        raise ValueError("仅支持灰度或单通道 tif 视频 (T, H, W) 或 (T, H, W, 1)")

    # 3. 保存
    tif.imwrite(
        out_path,
        filtered,
        bigtiff=False,
        metadata=meta,
        imagej=imagej
    )
    if verbose:
        print(f"[info] 已保存 -> {out_path}")


# ========== 滤波处理 ========== 
in_path = "./钙事件分割过程结果/含噪声影像_演示示例_预处理_背景剔除与归一化.tif"  # 输入16位TIFF视频
out_path = "./钙事件分割过程结果/含噪声影像_演示示例_预处理_高斯滤波.tif"  # 输出8位TIFF视频
gaussian_filter_tif_video(
    in_path=in_path,
    out_path=out_path,
    sigma=3)




#%% # # # # # # # # # # # # # # # # # # # # # # # # # # #  自适应钙活动分割  # # # # # # # # # # # # # # # # # # # # # # # # # # #  
from 自适应钙活动分割 import process_calcium_video  
input_video = "./钙事件分割过程结果/含噪声影像_演示示例_预处理_高斯滤波.tif"  # 输入16位TIFF视频
output_video = "./钙事件分割过程结果/含噪声影像_演示示例_自适应钙活动分割.tif"  # 输出8位TIFF视频

process_calcium_video(input_video, output_video, min_area=29, update_interval=5,n_sigma=2.5)

 






 

