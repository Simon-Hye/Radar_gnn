import numpy as np
import matplotlib.pyplot as plt
import os


def normalize_data(data):
    """
    对数据进行归一化（min-max归一化）
    参数:
        data: numpy 数组
    返回:
        归一化后的数据, 值域为 [0, 1]
    """
    min_val = np.min(data)
    max_val = np.max(data)
    # 防止除零
    if max_val - min_val == 0:
        return data
    return (data - min_val) / (max_val - min_val)


# 读取保存的 CSV 文件-
Zxx_real = np.loadtxt('processed_data/Zxx/Zxx_real.csv', delimiter=',')
Zxx_imag = np.loadtxt('processed_data/Zxx/Zxx_imag.csv', delimiter=',')
LBL = np.loadtxt('processed_data/Zxx/LBL.csv', delimiter=',')

# 将实部和虚部组合成复数
Zxx = Zxx_real + 1j * Zxx_imag  #(128, 14634)

# 1. 调整频谱顺序（将负频率部分移动到后面）
Zxx_swapped = np.concatenate((Zxx[64:-1], Zxx[0:63]), axis=0)

# 2. 计算幅值（取绝对值）
Zxx_magnitude = np.abs(Zxx_swapped)

# 3. 转换为 dB 形式
Zxx_db = 20 * np.log10(Zxx_magnitude + 1e-10)  # 避免 log(0) 错误

# 4. 计算对比度增强范围（去掉极端值，提高可视化效果）
vmin = np.percentile(Zxx_db, 5)   # 计算 5% 分位数
vmax = np.percentile(Zxx_db, 95)  # 计算 95% 分位数
Zxx_processed = np.clip(Zxx_db, vmin, vmax)  # 限制范围
Zxx_processed=normalize_data(Zxx_processed)  # 归一

# 5. 绘制处理后的 STFT 频谱图
plt.figure(figsize=(10, 6))
plt.imshow(Zxx_processed[:,0:1436], aspect='auto', cmap='jet')
plt.colorbar(label='Intensity (dB)')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Processed STFT Spectrogram")
plt.show()