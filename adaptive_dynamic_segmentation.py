import os
import numpy as np
from tifffile import TiffFile, TiffWriter
from sklearn.mixture import GaussianMixture
import dask.array as da
from tqdm import tqdm
from scipy.ndimage import label

DEFAULT_MIN_AREA = 30
HISTORY_FRAMES = 10

class TemporalMask:
    """时序一致性过滤：保留连续高亮像素"""
    def __init__(self, frames, min_duration=3, n_sigma=3.0):
        self.frames = frames
        self.min_d = min_duration
        self.n_sig = n_sigma
        self.h, self.w = frames[0].shape

    def _bg_thr(self, f):
        return np.mean(f) + self.n_sig * np.std(f)

    def run(self):
        masks = np.zeros((len(self.frames), self.h, self.w), dtype=bool)
        for t, f in enumerate(self.frames):
            masks[t] = f > self._bg_thr(f)

        valid = np.zeros((self.h, self.w), dtype=int)
        out = [np.zeros((self.h, self.w), dtype=bool) for _ in masks]
        for t, m in enumerate(masks):
            valid = np.where(m, valid + 1, 0)
            out[t] = valid >= self.min_d
        return out

class AdaptiveParams:
    """根据高亮像素比例 & 帧间漂移 自动更新 global / local / alpha"""
    def __init__(self, init_global=20, init_local=0.25, init_alpha=0.9):
        self.global_th = init_global
        self.local_th = init_local
        self.alpha = init_alpha
        self.history = []
        self.prev_mask = None

    def update(self, frame, mask):
        high_ratio = np.count_nonzero(mask) / mask.size
        self.history.append(high_ratio)
        if len(self.history) > HISTORY_FRAMES:
            self.history.pop(0)

        if len(self.history) >= 2:
            delta_ratio = abs(self.history[-1] - self.history[-2])
            self.global_th = max(self.global_th * (1.2 if delta_ratio < 0.001 else 0.8), 15 if delta_ratio >= 0.001 else 40)

            if self.prev_mask is not None:
                overlap = np.logical_and(mask, self.prev_mask).sum()
                union = np.logical_or(mask, self.prev_mask).sum()
                jaccard = overlap / (union + 1e-9)
                self.local_th = min(self.local_th * 1.3, 0.5) if jaccard > 0.7 else max(self.local_th * 0.9, 0.2)

        snr = np.mean(frame[mask]) / (np.std(frame[~mask]) + 1e-9)
        self.alpha = max(min(0.99, 1 - 1/snr), 0.85)

        self.prev_mask = mask.copy()
        return self.global_th, self.local_th, self.alpha

def adaptive_segments(frame, min_area):
    h, w = frame.shape
    X = frame.reshape(-1, 1)

    if np.unique(frame).size <= 1:
        ta = min_area
        tg = 10
        return [(frame.min(), frame.max(), tg, ta)]

    try:
        gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
    except Exception:
        gmm = GaussianMixture(n_components=1, random_state=0).fit(X)

    means = gmm.means_.flatten()
    idx = np.argsort(means)
    labels = gmm.predict(X).reshape(h, w)

    segs = []
    for k in range(len(means)):
        m = (labels == idx[k])
        if not np.any(m):
            continue
        ta = min_area if k == len(means) - 1 else max(min_area * 10, int(h * w * 0.05))
        tg = max(2, int(np.nan_to_num(np.std(frame[m])) * (0.5 if k == len(means) - 1 else 2)))
        low, high = frame[m].min(), frame[m].max()
        segs.append((low, high, tg, ta))
    segs.sort(key=lambda x: x[0])
    return segs or [(frame.min(), frame.max(), 10, min_area)]

def optimized_inverted_terrace_compress(frame, segs):
    h, w = frame.shape
    out = np.zeros_like(frame, dtype=np.uint8)  # 全黑初始化

    for low, high, tg, ta in reversed(segs):
        mask = (frame >= low) & (frame <= high)
        labeled_array, num_features = label(mask)

        for region_number in range(1, num_features + 1):
            region_mask = (labeled_array == region_number)
            if np.sum(region_mask) >= ta:
                # 钙事件区域保留信号值
                out[region_mask] = frame[region_mask]
          
    return out

def load_tiff_16(path):
    with TiffFile(path) as tif:
        return [p.asarray() for p in tif.pages]

def save_tiff_8(frames, path):
    with TiffWriter(path) as tw:
        for fr in frames:
            tw.write(fr.astype(np.uint8), photometric='minisblack')

def process_frame(fr, mask, min_area, prev_segs, update_interval, idx, alpha):
    fr_clean = fr.copy()

    if prev_segs is None or idx % max(1, int(update_interval * (1-alpha))) == 0:
        prev_segs = adaptive_segments(fr_clean, min_area)

    final_output = optimized_inverted_terrace_compress(fr_clean, prev_segs)
    
    # 未检测到事件的区域设置为黑色
    final_output[~mask] = 0

    return final_output

def process_calcium_video(input_video, output_video, min_area=DEFAULT_MIN_AREA, update_interval=35,n_sigma=3.0):
    frames_16 = load_tiff_16(input_video)
    masks = TemporalMask(frames_16, min_duration=3, n_sigma=n_sigma).run()
    
    adap = AdaptiveParams()
    out_frames = []
    prev_segs = None

    frames_da = da.from_array(frames_16, chunks=(1, frames_16[0].shape[0], frames_16[0].shape[1]))
    masks_da = da.from_array(masks, chunks=(1, masks[0].shape[0], masks[0].shape[1]))

    for idx in tqdm(range(len(frames_da)), desc="Processing frames"):
        fr = frames_da[idx].compute()
        mask = masks_da[idx].compute()

        g_thr, l_thr, alpha = adap.update(fr, mask)
        out_frame = process_frame(fr, mask, min_area, prev_segs, update_interval, idx, alpha)
        out_frames.append(out_frame)
    
    save_tiff_8(out_frames, output_video)
    print('Saved ->', output_video)
 