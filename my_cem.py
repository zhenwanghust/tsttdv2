import matplotlib.pyplot as plt
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import torch
from ts_generation import ts_generation
from Tools import standard
import torch.nn as nn

def cem(hsi_cube, target_spectrum):
    H, W, C = hsi_cube.shape
    data = hsi_cube.reshape(-1, C).T  # (C, N)
    N = data.shape[1]

    # 计算自相关矩阵
    R = (data @ data.T) / N  # 修正为自相关矩阵

    # 计算 CEM 滤波器权重
    R_inv = np.linalg.pinv(R)
    denominator = target_spectrum.T @ R_inv @ target_spectrum
    w = (R_inv @ target_spectrum) / denominator

    # 计算响应
    response = w.T @ data
    return response.reshape(H, W)
#
# def cem(hsi_cube, target_spectrum):
#     H, W, C = hsi_cube.shape
#     data = hsi_cube.reshape(-1, C).T  # (C, N)
#     target_spectrum = target_spectrum.flatten()
#
#     mean_spectrum = np.mean(data, axis=1, keepdims=True)
#     data_centered = data - mean_spectrum
#     target_centered = target_spectrum - mean_spectrum.flatten()
#
#     num = np.sum(target_centered[:, None] * data_centered, axis=0)
#     denom = np.sqrt(np.sum(target_centered ** 2) * np.sum(data_centered ** 2, axis=0))
#     response = num / denom
#     return response.reshape(H, W)
#

if __name__ == '__main__':
    path = 'Sandiego.mat'
    mat = sio.loadmat(path)
    hsi_cube = mat['data']
    gt = mat['map']
    hsi_cube = standard(hsi_cube)
    target_spectrum = ts_generation(hsi_cube, gt, 7)
    result = cem(hsi_cube, target_spectrum)

