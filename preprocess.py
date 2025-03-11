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


II, JJ, KK = np.shape(RT) # II=range bins数量，JJ=slow-time bins数量，KK=radar nodes数量


fs=1/ts
time_Start=0
time_End=JJ-1
window_length=128
overlap_length=127
signal = np.sum(RT[:-1,time_Start:time_End, 0],axis=0) #加上最后一个range bin后有很大的噪声

f, t, Zxx = stft(signal, fs=fs, nperseg=window_length, noverlap=overlap_length,return_onesided=False)

# 将实部和虚部分别提取出来
real_part = Zxx.real
imag_part = Zxx.imag

directory = 'processed_data/Zxx/'

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the real and imaginary parts as CSV files
np.savetxt(os.path.join(directory, 'Zxx_real.csv'), real_part, fmt='%.4f', delimiter=',')
np.savetxt(os.path.join(directory, 'Zxx_imag.csv'), imag_part, fmt='%.4f', delimiter=',')

# Save the labels as a CSV file
np.savetxt(os.path.join(directory, 'LBL.csv'), LBL, fmt='%d', delimiter=',')

#  循环处理所有数据(所有人，五个雷达节点)：























