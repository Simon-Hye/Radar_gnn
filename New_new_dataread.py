import scipy.io
import os
import numpy as np
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

# def data_stft(signal, fs, window_duration, overlap_percentage):
#     window_length = int(fs * (window_duration * 1e-3))
#     overlap_samples = int(window_length * (overlap_percentage * 1e-2))
#     f, t, Zxx = stft(signal, fs=fs, nperseg=window_length, noverlap=overlap_samples,return_onesided=False)
#
#     # 将强度转换为dB
#     intensity_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
#     f = np.around(f, decimals=1)
#     max_intensity_indices = np.argmax(intensity_db, axis=0)
#     max_frequencies = f[max_intensity_indices]
#     return max_frequencies


# def plot_stft_2(signal, fs, window_length, overlap_length):
#     f, t, Zxx = stft(signal, fs=fs, nperseg=window_length, noverlap=overlap_length)
#     intensity_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
#     #intensity_db =20*np.log10(np.abs(Zxx))
#
#     #202503062329修改
#     # Zxx的形状是（M,N)，M是频率数，N是时间窗数
#     #将Zxx的实部和虚部分离
#     # M, N= Zxx.shape
#     # Zxx_real=np.real(Zxx)
#     # Zxx_imag=np.imag(Zxx)
#
#     #组合成（M，N，2）的数组
#     # Zxx_separated=np.stack((Zxx_real,Zxx_imag),axis=-1)
#     # print(Zxx.shape)
#     # print(Zxx_separated.shape)
#     # print(M,N)
#     print(np.abs(Zxx))
#     # print(Zxx_separated)
#     # plt.figure(figsize=(10, 5))
#     # plt.subplot(1, 2, 1)
#     #plt.pcolormesh(t, f, np.abs(Zxx),shading='auto')
#     #plt.pcolormesh(t, f, Zxx,shading='auto')
#     #plt.pcolormesh(t, f, intensity_db,vmin=36,shading='auto')
#     # plt.pcolormesh(t, f, np.abs(Zxx),vmin=0,vmax=500,shading='auto')
#     #plt.imshow(intensity_db, aspect='auto', cmap='jet',extent=[0, ts * JJ, 1, window_length])
#     x = np.linspace(0, ts * JJ, intensity_db.shape[1])
#     y = np.linspace(1, window_length, intensity_db.shape[0])
#     plt.pcolormesh(x, y, intensity_db,shading='auto',cmap='jet')
#     # plt.pcolormesh(t, f, intensity_db, shading='auto', cmap='jet')
#     plt.colorbar(label='Intensity [dB]')
#     plt.title(f"Spectrogram (Window Length: {window_length}ms, Overlap_length: {overlap_length})")
#     plt.ylabel('Doppler')
#     plt.xlabel('Time [sec]')
#     plt.tight_layout()
#     plt.show()

fs=1/ts
time_Start=400
time_End=1600
window_length=128
overlap_length=127
signal = np.sum(RT[:-1,time_Start:time_End, 0],axis=0) #加上最后一个range bin后有很大的噪声
#plot_stft_2(signal=signal_sum, fs=fs, window_length=128,overlap_length=127)
# SPC= stft(signal, fs=fs, nperseg=window_length, noverlap=overlap_length)
# SPC_new=np.concatenate(SPC[64:-1],SPC[0:63]) #此处报错
# f, t, Zxx =SPC_new

f, t, Zxx = stft(signal, fs=fs, nperseg=window_length, noverlap=overlap_length)

# 调换上下两部分
Zxx_swapped = np.concatenate((Zxx[64:-1], Zxx[0:63]), axis=0)  # 调换频谱矩阵
f_swapped = np.concatenate((f[64:-1], f[0:63]))  # 调换频率轴

intensity_db = 20 * np.log10(np.abs(Zxx_swapped) + 1e-10)
print(np.abs(Zxx))
# x = np.linspace(0, ts * JJ, intensity_db.shape[1])
# y = np.linspace(1, window_length, intensity_db.shape[0])
# plt.pcolormesh(x, y, intensity_db, shading='auto', cmap='jet') #要求输入的x,y单调
# 色彩一
vmin = np.percentile(intensity_db, 5)  # 5% 分位数
vmax = np.percentile(intensity_db, 95)  # 95% 分位数
plt.imshow(intensity_db, aspect='auto', cmap='jet',vmin=vmin,vmax=vmax,
           extent=[0, ts * JJ, 1, window_length])

# 色彩二（正常）
# plt.imshow(intensity_db, aspect='auto', cmap='jet',
#            extent=[0, ts * JJ, 1, window_length])
# plt.pcolormesh(t, f, intensity_db, shading='auto', cmap='jet')
# plt.ylim(f.min(), f.max())  # 明确设置频率轴范围
plt.colorbar(label='Intensity (dB)')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("STFT Spectrogram")
# plt.show()


# plt.pcolormesh(t, f, intensity_db, shading='auto', cmap='viridis')

# plt.colorbar(label='Intensity [dB]')
# #plt.title(f"Spectrogram (Window Length: {window_length}ms, Overlap_length: {overlap_length})")
# plt.xlabel('Time [sec]')
# plt.ylabel('Doppler')


##打印类别描述##
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
# time_samples = np.linspace(0, ts * np.size(LBL), np.size(LBL))
time_samples = np.linspace(time_Start, time_End, np.size(LBL))
plt.plot(time_samples, LBL.flatten(), 'red')

LBL_flat = LBL.flatten()
print("LBL.flatten()的具体数值:", LBL_flat)

# *0.0082 转换成秒
label_matrix=np.zeros((1,10))
cnt=0
for i in range(300,400):
        cnt+=1# print(LBL_flat[i*20],end=" ")
        label_matrix[0,LBL_flat[i*20]]=1
        if(cnt==10):
            print(label_matrix)
            label_matrix=np.zeros((1,10))
            cnt=0

plt.ylim([-1, 10])
plt.xlabel("slowtime (sec)")
plt.ylabel("Classes")

plt.show()

# print("t shape:", t.shape)  # 时间轴
# print("f shape:", f.shape)  # 频率轴
# print("intensity_db shape:", intensity_db.shape)  # 数据矩阵

import numpy as np

# # 检查数据中是否有 NaN 或 Inf
# if np.isnan(intensity_db).any() or np.isinf(intensity_db).any():
#     print("数据中存在 NaN 或 Inf 值")
#     # 替换 NaN 或 Inf 值
#     intensity_db = np.nan_to_num(intensity_db, nan=0.0, posinf=0.0, neginf=0.0)

# # 检查 t 和 f 的范围
# print("t range:", t.min(), t.max())
# print("f range:", f.min(), f.max())

# # 绘图时确保范围正确
# plt.pcolormesh(t, f, intensity_db, shading='auto', cmap='jet')
# plt.colorbar(label='Intensity (dB)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.title("STFT Spectrogram")
# plt.show()


# vmin = np.percentile(intensity_db, 5)  # 5% 分位数
# vmax = np.percentile(intensity_db, 95)  # 95% 分位数
#
# plt.pcolormesh(t, f, intensity_db, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
# plt.colorbar(label='Intensity (dB)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.title("STFT Spectrogram with Adjusted Color Range")
# plt.show()

# # 绘制负频率部分
# negative_f_indices = np.where(f < 0)[0]
# negative_f = f[negative_f_indices]
# negative_intensity_db = intensity_db[negative_f_indices, :]
#
# plt.figure()
# plt.pcolormesh(t, negative_f, negative_intensity_db, shading='auto', cmap='jet')
# plt.colorbar(label='Intensity (dB)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.title("Negative Frequencies")
# plt.show()
#
# # 绘制正频率部分
# positive_f_indices = np.where(f >= 0)[0]
# positive_f = f[positive_f_indices]
# positive_intensity_db = intensity_db[positive_f_indices, :]
#
# plt.figure()
# plt.pcolormesh(t, positive_f, positive_intensity_db, shading='auto', cmap='jet')
# plt.colorbar(label='Intensity (dB)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.title("Positive Frequencies")
# plt.show()

# print("Negative Frequencies Stats:")
# print("Min:", np.min(intensity_db[f < 0]))
# print("Max:", np.max(intensity_db[f < 0]))
# print("Mean:", np.mean(intensity_db[f < 0]))
#
# print("Positive Frequencies Stats:")
# print("Min:", np.min(intensity_db[f >= 0]))
# print("Max:", np.max(intensity_db[f >= 0]))
# print("Mean:", np.mean(intensity_db[f >= 0]))

# # 确保频率轴是升序排列
# f_sorted_indices = np.argsort(f)
# f_sorted = f[f_sorted_indices]
# intensity_db_sorted = intensity_db[f_sorted_indices, :]
# # 绘制排序后的数据
# plt.pcolormesh(t, f_sorted, intensity_db_sorted, shading='auto', cmap='jet')
# plt.colorbar(label='Intensity (dB)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.title("STFT Spectrogram (Sorted Frequencies)")
# plt.show()