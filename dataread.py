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
ts = 0.0082  # 采样时间间隔（秒）
range_0 = 1  # 最小范围（米）
range_max = 4.8  # 最大范围（米）

# 加载.mat文件
mat_struct = scipy.io.loadmat(path)
RT = mat_struct['hil_resha_aligned']  # 雷达数据
LBL = mat_struct['lbl_out']  # 标签数据


II, JJ, KK = np.shape(RT)
print(f"Slow-time bins数量：\t{JJ}\nRange bins数量：\t{II}\nRadar nodes数量：\t{KK}")

def data_stft(signal, fs, window_duration, overlap_percentage):
    window_length = int(fs * (window_duration * 1e-3))
    overlap_samples = int(window_length * (overlap_percentage * 1e-2))
    f, t, Zxx = stft(signal, fs=fs, nperseg=window_length, noverlap=overlap_samples)
    # 将强度转换为dB
    intensity_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    f = np.around(f, decimals=1)
    max_intensity_indices = np.argmax(intensity_db, axis=0)
    max_frequencies = f[max_intensity_indices]
    return max_frequencies

    # plt.pcolormesh(t, f, intensity_db,vmin=30)
    # plt.colorbar(label='Intensity [dB]')
    # plt.title(f"Spectrogram (Window Duration: {window_duration}ms, Overlap: {overlap_percentage}%)")
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

fs=1/ts
# for i in range(5):
#     signal = np.sum(RT[:, 6000:8000, i],axis=0)#这里现在取了一个运动的部分（每个数据集可以先看一下整体的时间距离图像，再对RT的第一二维度选择合适的范围）
# ####signal = RT[2, 3400:3800, 0]
# ####选取了雷达数据中的第一行（range靠近雷达的静止处），时间从3400到3800（脉冲时间），第一个通道
#     plot_stft(signal, fs, window_duration=256, overlap_percentage=75)#窗口时间越长使得时间带越宽；重叠率越高使得频率分辨率越高（图像更连续）


# 此处输出固定时间下最大强度出对应的频率
time_Start=6000
time_End=8000
signal = np.sum(RT[:,time_Start:time_End, 0],axis=0)
print(data_stft(signal, fs, window_duration=256, overlap_percentage=50))
# signal = np.sum(RT[:, time_Start:time_End, 0],axis=0)#这里现在取了一个运动的部分（每个数据集可以先看一下整体的时间距离图像，再对RT的第一二维度选择合适的范围）
# plot_stft(signal, fs, window_duration=256, overlap_percentage=75)#窗口时间越长使得时间带越宽；重叠率越高使得频率分辨率越高（图像更连续）

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

plt.tight_layout()
plt.show()