import scipy.io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft

# 输入.mat文件路径
path = 'path/to/data/1/001_mon_wal_1.mat'
while not os.path.isfile(path):
    print("请输入.mat文件的绝对路径：")
    path = os.path.normpath(input())
    if not os.path.isfile(path):
        print(f"路径 {path} 无效，请重新输入。")

# 雷达参数（不修改）
ts = 0.0082
range_0 = 1
range_max = 4.8

# 加载.mat文件
mat_struct = scipy.io.loadmat(path)
RT = mat_struct['hil_resha_aligned']  # 雷达数据
LBL = mat_struct['lbl_out']  # 标签数据

II, JJ, KK = np.shape(RT)
print(f"Slow-time bins数量：\t{JJ}\nRange bins数量：\t{II}\nRadar nodes数量：\t{KK}")
N = 5
time_Start = 0
time_End = JJ - 1
signal2 = np.sum(RT[:-1, time_Start:time_End, 0], axis=0) 


##range-time
# intensity_db=20*np.log10(np.abs(RT[:-1,:,0]), where=np.abs(RT[:-1,:, 0]) > 0)
# extent = [range_0, range_max,0, ts * JJ]
# fig, ax = plt.subplots(figsize=(10, 6))
# im = ax.imshow(intensity_db, cmap='jet', origin='lower', extent=[0,ts*JJ,range_0,range_max])
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Range (m)')
# ax.set_title('Range-Time Matrix')
# plt.colorbar(im, ax=ax, label='Intensity (dB)')
# plt.tight_layout()
# plt.show()

fs = 1 / ts
window_length = 128
overlap_length = 127

# 对signal2进行stft
f, t, Zxx = stft(signal2, fs=fs, nperseg=window_length, noverlap=overlap_length,return_onesided=False)
Zxx_swapped = np.concatenate((Zxx[64:-1], Zxx[0:63]), axis=0)  # 调换频谱矩阵
f_swapped = np.concatenate((f[64:-1], f[0:63]))  # 调换频率轴
intensity_db = 20 * np.log10(np.abs(Zxx_swapped) + 1e-10)

# 绘图等操作
vmin = np.percentile(intensity_db, 5)  # 5% 分位数
vmax = np.percentile(intensity_db, 95)  # 95% 分位数
plt.imshow(intensity_db, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax,
           extent=[time_Start*ts, ts * time_End, 1, window_length])
plt.colorbar(label='Intensity (dB)')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("STFT Spectrogram")


print("complete")

# Plot labels
plt.figure(figsize=(10, 5))
time_samples = np.linspace(time_Start, ts * time_End, np.size(LBL))
plt.plot(time_samples, LBL.flatten(), 'red')
plt.ylim([-1, 10])
plt.xlabel("slowtime (sec)")
plt.ylabel("Classes")
plt.show()