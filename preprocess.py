import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft



#  循环处理所有数据(所有人，五个雷达节点)：


# 指定路径
root_path = r"D:\YZ\Ronny_MonostaticData"

# 遍历根路径下的所有顶级文件夹
for top_folder_name in os.listdir(root_path):
    top_folder_path = os.path.join(root_path, top_folder_name)
    if os.path.isdir(top_folder_path):  # 确保是文件夹
        # 假设每个顶级文件夹中只有一个子文件夹（这里取第一个遇到的子文件夹）
        sub_folders = [f for f in os.listdir(top_folder_path) if os.path.isdir(os.path.join(top_folder_path, f))]
        if sub_folders:  # 确保存在子文件夹
            sub_folder_path = os.path.join(top_folder_path, sub_folders[0])

            # 获取该子文件夹下所有文件的路径和文件名，并按名称升序排序
            files = sorted(
                [(os.path.join(sub_folder_path, f), os.path.basename(f)) for f in os.listdir(sub_folder_path) if
                 os.path.isfile(os.path.join(sub_folder_path, f))])

            # 取前28个文件（如果文件少于28个，则打印所有）
            first_28_files = files[:28]

            # 打印前28个文件的路径和文件名
            for file_path, file_name in first_28_files:
                print(f"路径: {file_path}, 文件名: {file_name}")
                # 输入.mat文件路径
                path = file_path
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

                II, JJ, KK = np.shape(RT)  # II=range bins数量，JJ=slow-time bins数量，KK=radar nodes数量

                fs = 1 / ts
                time_Start = 0
                time_End = JJ - 1
                window_length = 128
                overlap_length = 127

                for radar_num in range(5):
                    signal = np.sum(RT[:-1, time_Start:time_End, radar_num], axis=0)  # 加上最后一个range bin后有很大的噪声

                    f, t, Zxx = stft(signal, fs=fs, nperseg=window_length, noverlap=overlap_length,
                                     return_onesided=False)

                    # 将实部和虚部分别提取出来
                    real_part = Zxx.real
                    imag_part = Zxx.imag

                    directory_base = 'processed_data/Zxx/' + file_name + '/'

                    # Create the directory if it doesn't exist
                    if not os.path.exists(directory_base):
                        os.makedirs(directory_base)
                    directory = os.path.join(directory_base, f'radar_{radar_num}')
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # Save the real and imaginary parts as CSV files
                    np.savetxt(os.path.join(directory, 'Zxx_real.csv'), real_part, fmt='%.4f', delimiter=',')
                    np.savetxt(os.path.join(directory, 'Zxx_imag.csv'), imag_part, fmt='%.4f', delimiter=',')

                    # Save the labels as a CSV file
                    np.savetxt(os.path.join(directory, 'LBL.csv'), LBL, fmt='%d', delimiter=',')





















