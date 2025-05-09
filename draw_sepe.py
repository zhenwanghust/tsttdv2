import matplotlib.pyplot as plt
import numpy as np
import os
# 此代码没有使用
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模拟数据：真实情况请替换这里的数据
np.random.seed(42)
data = {
    'SAM': {'target': np.random.uniform(0.5, 0.8, 100), 'background': np.random.uniform(0.1, 0.6, 100)},
    'SID': {'target': np.random.uniform(0.4, 0.7, 100), 'background': np.random.uniform(0.05, 0.5, 100)},
    'ACE': {'target': np.random.uniform(0.1, 0.4, 100), 'background': np.random.uniform(0.0, 0.2, 100)},
    'CEM': {'target': np.random.uniform(0.3, 0.8, 100), 'background': np.random.uniform(0.2, 0.5, 100)},
    'MF': {'target': np.random.uniform(0.7, 1.0, 100), 'background': np.random.uniform(0.3, 0.6, 100)},
    'SFCTD': {'target': np.random.uniform(0.9, 1.0, 100), 'background': np.random.uniform(0.2, 0.6, 100)},
    'TSCNTD': {'target': np.random.uniform(0.85, 1.0, 100), 'background': np.random.uniform(0.1, 0.6, 100)},
    'ours': {'target': np.random.uniform(0.98, 1.0, 100), 'background': np.random.uniform(0.0, 0.02, 100)}
}
# data = {
#     'SAM': {'target': SAM_predict[label_map == 1], 'background': SAM_predict[label_map == 0]},
#     'SID': {'target': SID_predict[label_map == 1], 'background': SID_predict[label_map == 0]},
#     'ACE': {'target': ACE_predict[label_map == 1], 'background': ACE_predict[label_map == 0]},
#     # 以此类推...
# }


fig, ax = plt.subplots(figsize=(10, 6))

positions = []
data_to_plot = []
colors = []

# 组装数据，交替排列目标和背景
for i, method in enumerate(data.keys()):
    target_values = data[method]['target']
    background_values = data[method]['background']

    positions.append(i * 2 + 1)
    data_to_plot.append(target_values)
    colors.append('red')

    positions.append(i * 2 + 2)
    data_to_plot.append(background_values)
    colors.append('blue')

# 绘制箱型图
bp = ax.boxplot(
    data_to_plot,
    positions=positions,
    widths=0.6,
    patch_artist=True,
    showfliers=False  # 不显示离群点
)

# 设置颜色
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# 设置x轴标签
ax.set_xticks([(i * 2 + 1.5) for i in range(len(data))])
ax.set_xticklabels(list(data.keys()), fontsize=10)

# 设置中文纵轴标题
ax.set_ylabel("检测值归一化结果", fontsize=12)

# 添加标题，替换XX数据集为你的真实数据集名
ax.set_title("基于XX数据集的检测算法可分性分析", fontsize=14)

# 图例
target_patch = plt.Line2D([0], [0], color='red', lw=4, label='目标')
background_patch = plt.Line2D([0], [0], color='blue', lw=4, label='背景')
ax.legend(handles=[target_patch, background_patch], loc='upper left')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()
save_dir = "picture/sepe"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"mock.png")
plt.savefig(save_path)
print("save picture to {}".format(save_path))