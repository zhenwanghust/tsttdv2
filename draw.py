import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def standard(data):
    """标准化数据到 [0, 1]"""
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data


def save_image(image, save_path, cmap=None):
    """保存图像到指定路径"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, image, cmap=cmap)


def visualize_hsi(data, gt, filename):
    """分别保存高光谱伪彩色图像和Ground Truth"""
    # 选择三个波段作为 RGB 显示（假设 data 形状为 [H, W, C]）
    h, w, c = data.shape
    bands = [12, 66, 50]  # 可调整的波段索引

    # 确保所选波段在范围内
    bands = [min(c - 1, b) for b in bands]

    rgb_image = np.stack([data[:, :, bands[0]], data[:, :, bands[1]], data[:, :, bands[2]]], axis=-1)
    rgb_image = (rgb_image * 255).astype(np.uint8)  # 转换为 8-bit 图像

    # 定义保存路径
    falsecolor_path = f"picture/result/falsecolor/{filename}.png"
    gt_path = f"picture/result/gt/{filename}.png"

    # 保存图像
    save_image(rgb_image, falsecolor_path)
    save_image(gt, gt_path, cmap='gray')

    print(f"Saved false color image to {falsecolor_path}")
    print(f"Saved ground truth image to {gt_path}")



def draw_ROC():

    methods = ['CEM', 'ACE', 'MF', 'MLSN', 'HTD-IRN', 'TSTTD', 'Proposed']
    colors = ['r', 'g', 'b', 'k', 'm', 'c', 'y']
    linestyles = ['--', '--', '--', '--', '--', '--','--']

    plt.figure()
    for method, color, linestyle in zip(methods, colors, linestyles):
        data = np.load(f'{method}.npz')
        plt.plot(data['fpr'], data['tpr'], color=color, linestyle=linestyle, label=method)

    plt.xlabel('False Alarm Rate')
    plt.ylabel('Probability of Detection')
    plt.legend()
    plt.grid()

    # 保存图像
    save_image(rgb_image, falsecolor_path)

    print(f"Saved false color image to {falsecolor_path}")


if __name__ == '__main__':
    # 读取 .mat 文件
    path = "Indian_pines_corrected.mat"  # 替换为实际路径
    filename = os.path.splitext(os.path.basename(path))[0]  # 获取文件名（无扩展名）
    # mat = sio.loadmat(path)
    # data = standard(mat['data'])
    # gt = mat['map']
    mat = sio.loadmat(path)
    data = mat['indian_pines_corrected']
    mat1 = sio.loadmat("Indian_pines_gt.mat")
    temp_gt = mat1['indian_pines_gt']
    gt = (temp_gt == 16).astype(int)
    # 可视化并保存
    visualize_hsi(data, gt, filename)

    # 画出ROC文件
    draw_ROC()