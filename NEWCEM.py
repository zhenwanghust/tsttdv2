import matplotlib.pyplot as plt
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import torch
from ts_generation import ts_generation
from Tools import standard
import torch.nn as nn
from sklearn import metrics
import os

def save_image(image, save_path, cmap=None):
    """保存图像到指定路径"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, image, cmap=cmap)

def cem(hsi_cube, target_spectrum):
    H, W, C = hsi_cube.shape
    data = hsi_cube.reshape(-1, C).T  # (C, N)
    N = data.shape[1]

    R = (data @ data.T) / N  # 自相关矩阵
    R_inv = np.linalg.pinv(R)
    denominator = target_spectrum.T @ R_inv @ target_spectrum
    w = (R_inv @ target_spectrum) / denominator
    response = w.T @ data
    return response.reshape(H, W)


def ace(hsi_cube, target_spectrum):
    H, W, C = hsi_cube.shape
    data = hsi_cube.reshape(-1, C).T  # (C, N)
    target_spectrum = target_spectrum.flatten()

    mean_spectrum = np.mean(data, axis=1, keepdims=True)
    data_centered = data - mean_spectrum
    target_centered = target_spectrum - mean_spectrum.flatten()

    num = np.sum(target_centered[:, None] * data_centered, axis=0)
    denom = np.sqrt(np.sum(target_centered ** 2) * np.sum(data_centered ** 2, axis=0))
    response = num / denom
    return response.reshape(H, W)


def mf(hsi_cube, target_spectrum):
    H, W, C = hsi_cube.shape
    data = hsi_cube.reshape(-1, C).T  # (C, N)
    N = data.shape[1]

    mean_spectrum = np.mean(data, axis=1, keepdims=True)
    data_centered = data - mean_spectrum
    target_spectrum = target_spectrum.flatten() - mean_spectrum.flatten()

    R_hat = (data_centered @ data_centered.T) / (N - 1)
    R_inv = np.linalg.pinv(R_hat)
    tmp = target_spectrum.T @ R_inv @ target_spectrum
    response = np.array([(x.T @ R_inv @ target_spectrum) / tmp for x in data_centered.T])

    return response.reshape(H, W)


def compute_auc(result, gt, path,method,epsilon=5):
    y_l = np.reshape(gt, [-1, 1], order='F')
    y_p = np.reshape(result, [-1, 1], order='F')
    fpr, tpr, thresholds = metrics.roc_curve(y_l, y_p, drop_intermediate=False)

    fpr = fpr[1:]
    tpr = tpr[1:]
    thresholds = thresholds[1:]
    save_path = f"csv/{path}/{method}.npz"
    # print(len(fpr), len(tpr), len(thresholds))

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path,
             thresholds=thresholds,
             pd=tpr,
             far=fpr)
    # 保存 npz 文件
    # print(f"Saved npz file to {save_path}")

    auc1 = round(metrics.auc(fpr, tpr), epsilon)
    auc2 = round(metrics.auc(thresholds, fpr), epsilon)
    auc3 = round(metrics.auc(thresholds, tpr), epsilon)
    auc4 = round(auc1 + auc3 - auc2, epsilon)
    auc5 = round(auc3 / auc2, epsilon)

    # 2a. 方法一：用 precision_recall_curve + auc 计算 PR-AUC
    precision, recall, pr_thresholds = metrics.precision_recall_curve(y_l, y_p)
    # precision_recall_curve 会多返回一个触发最低 recall=1.0 的点，通常直接用 recall,precision
    pr_auc1 = round(metrics.auc(recall, precision), epsilon)

    # 2b. 方法二：直接用 average_precision_score
    # average_precision_score 本质也是对 PR 曲线积分，通常更推荐
    pr_auc2 = round(metrics.average_precision_score(y_l, y_p), epsilon)

    # print(f"ROC-AUC = {auc1}")
    # print(f"PR-AUC (via auc) = {pr_auc1}")
    # print(f"PR-AUC (via average_precision_score) = {pr_auc2}")

    return pr_auc2, auc2, auc3, auc4, auc5


def visualize_results(result, title):
    plt.figure()
    plt.imshow(result, cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    path = 'Sandiego2.mat'#urban1
    print(path)
    if path == "Indian_pines_corrected.mat":
        mat = sio.loadmat(path)
        hsi_cube = mat['indian_pines_corrected']
        mat1 = sio.loadmat("Indian_pines_gt.mat")
        temp_gt = mat1['indian_pines_gt']
        gt = (temp_gt == 16).astype(int)
    else:
        mat = sio.loadmat(path)
        hsi_cube = mat['data']
        gt = mat['map']

    hsi_cube = standard(hsi_cube)
    h,w,c = hsi_cube.shape
    target_spectrum = ts_generation(hsi_cube, gt, 7)

    for method, func in zip(["CEM", "ACE", "MF"], [cem, ace, mf]):
        result = func(hsi_cube, target_spectrum)
        detection_map = np.reshape(result, [h, w], order='F')
        detection_map = standard(detection_map)
        # savepath = f"picture/result/{method}/{path}.png"
        #
        # # 保存图像
        # save_image(detection_map, savepath)
        # result = np.clip(detection_map, 0, 1)


        pr, auc2, auc3, auc4, auc5 = compute_auc(detection_map, gt,path,method)
        # print(f"{method} AUC1: {auc1:.4f}, AUC2: {auc2:.4f}, AUC3: {auc3:.4f}, AUC4: {auc4:.4f}, AUC5: {auc5:.4f}")
        print(f"{method} pr: {pr:.4f}")

        # visualize_results(result, f"{method} Detection Result")
