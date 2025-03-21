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
signal2 = []
distance = np.zeros(JJ)
for i in range(JJ): #时间
    ax = 0
    label = 0
    previous_distances = []
    for j in range(II): #距离格
        temp = RT[j, i, 0]
        if np.abs(temp) > ax:
            ax = np.abs(temp)
            label = j
        previous_distances.append(RT[j, i, 0])
#####在这个循环里写进signal2的数据。
    if j+N<II and j-N>=0:
        signal2.append(np.sum(RT[j-N:j+N,i,0],axis=0))
    elif j+N>=II:
        signal2.append(np.sum(RT[II-2*N:II-1,i,0],axis=0))
    elif j-N<0:
        signal2.append(np.sum(RT[0:2*N,i,0],axis=0))
    # np.append(signal2,np.sum(RT[j-N:j+N,i,0],axis=0))

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
time_Start =400
time_End = 1600
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
plt.title(f"STFT Spectrogram N={N}")

# 打印类别描述
class_descriptions = (
    "The classes are:\n"
    "0 Idle\n"
    "1 Walking\n"
    "2 Nothing (stationary)\n"
    "3 Sitting down\n"
    "4 Standing up from sitting\n"
    "5 Bending from Sitting\n"
    "6 Bending from Standing\n"
    "7 Falling from Walking\n"
    "8 Standing up from the ground\n"
    "9 Falling from Standing\n"
)
print(class_descriptions)

# Plot labels
plt.figure(figsize=(10, 5))
time_samples = np.linspace(time_Start, time_End, np.size(LBL))
plt.plot(time_samples, LBL.flatten(), 'red')
plt.ylim([-1, 10])
plt.xlabel("slowtime (sec)")
plt.ylabel("Classes")
plt.show()