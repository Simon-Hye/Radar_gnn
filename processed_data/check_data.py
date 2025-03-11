import numpy as np
import matplotlib.pyplot as plt

# 读取保存的 CSV 文件
Zxx_real = np.loadtxt('processed_data/Zxx/Zxx_real.csv', delimiter=',')
Zxx_imag = np.loadtxt('processed_data/Zxx/Zxx_imag.csv', delimiter=',')
LBL = np.loadtxt('processed_data/Zxx/LBL.csv', delimiter=',')

# 将实部和虚部组合成复数
Zxx = Zxx_real + 1j * Zxx_imag  #(128, 14634)



