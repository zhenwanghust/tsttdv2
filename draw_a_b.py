import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 确保保存目录存在
save_dir = "picture/abparamselection"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "3d_plot-sandiego2-0417.png")
def draw_3d(Z):
    # 生成假数据
    # 定义 alpha 和 beta
    alpha = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55])
    beta = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55])
    M, E = np.meshgrid(alpha, beta)
    # 创建 3D 图像
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面图
    surf = ax.plot_surface(M, E, Z, cmap='autumn', edgecolor='k')

    # 添加数据标签
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    ax.text(M[max_idx], E[max_idx], Z[max_idx], f"{Z[max_idx]:.4f}", color='blue', fontsize=8)

    # 轴标签
    ax.set_xlabel(r'$Alpha$', fontsize=8)
    ax.set_ylabel(r'$Beta$', fontsize=8)
    ax.set_zlabel(r'AUC', fontsize=8)
    ax.tick_params(labelsize=6)
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight',pad_inches=0.3)
    plt.close()

    print(f"图片已保存到 {save_path}")

if __name__ == '__main__':
    #实验
    Z = np.array([
        [0.99697, 0.99723, 0.99751, 0.99743, 0.99758, 0.99774, 0.99786, 0.99817, 0.99792, 0.99834, 0.99832],
        [0.99792, 0.99799, 0.99815, 0.99835, 0.99834, 0.99837, 0.99869, 0.99862, 0.99860, 0.99871, 0.99857],
        [0.99858, 0.99869, 0.99875, 0.99867, 0.99878, 0.99882, 0.99889, 0.99884, 0.99884, 0.99879, 0.99864],
        [0.99888, 0.99893, 0.99895, 0.99896, 0.99903, 0.99903, 0.99901, 0.99897, 0.99893, 0.99878, 0.99861],
        [0.99897, 0.99910, 0.99912, 0.99914, 0.99906, 0.99901, 0.99899, 0.99897, 0.99867, 0.99857, 0.99856],
        [0.99916, 0.99913, 0.99914, 0.99907, 0.99909, 0.99903, 0.99898, 0.99882, 0.99870, 0.99844, 0.99844],
        [0.99911, 0.99911, 0.99913, 0.99904, 0.99897, 0.99890, 0.99868, 0.99855, 0.99845, 0.99828, 0.99836],
        [0.99910, 0.99909, 0.99901, 0.99899, 0.99888, 0.99871, 0.99853, 0.99844, 0.99831, 0.99827, 0.99844],
        [0.99907, 0.99899, 0.99896, 0.99891, 0.99877, 0.99851, 0.99836, 0.99835, 0.99837, 0.99838, 0.99840],
        [0.99914, 0.99909, 0.99903, 0.99911, 0.99876, 0.99833, 0.99807, 0.99817, 0.99824, 0.99841, 0.99843],
        [0.99910, 0.99891, 0.99897, 0.99891, 0.99836, 0.99752, 0.99775, 0.99790, 0.99844, 0.99870, 0.99857]
    ])



    draw_3d(Z)