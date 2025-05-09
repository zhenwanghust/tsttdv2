import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

# 方法名
methods = ['CEM', 'ACE', 'MF', 'MLSN', 'TSTTD', 'HTDIRN', 'Proposed']
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
linestyles = ['-', '--', '-.', ':', '-', '--', '-']
# methods = ['CEM', 'ACE', 'MF']
#
# colors = ['r', 'g', 'b']
# linestyles = ['-', '--', '-.']
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 模拟数据生成函数（真实使用中请替换为你的检测结果）
def generate_mock_data():
    thresholds = np.linspace(0, 1, 50)
    pd = [np.clip(np.random.rand(50).cumsum()/50, 0, 1) for _ in methods]
    far = [np.clip(np.sort(np.random.rand(50)), 0, 1) for _ in methods]
    return thresholds, pd, far
def uniform_sample(array, target_len=2550):
    """
    对数组进行均匀采样，返回长度为 target_len 的新数组
    """
    original_len = len(array)
    if original_len == target_len:
        return array
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, array)

def load_real_data_from_npz(base_dir, methods):
    """
    从多个方法的 .npz 文件中加载 pd, far, thresholds，整合成画图所需格式
    :param base_dir: 所有 .npz 文件所在的目录
    :param methods: 方法名称列表，对应的文件为 base_dir/{method}.npz
    :return: thresholds, pd_list, far_list
    """
    pd_list = []
    far_list = []
    thresholds = None
    for method in methods:
        file_path = os.path.join(base_dir, f"{method}.npz")
        data = np.load(file_path)
        # print(method)
        # print(len(data['pd']))  # 查看原始长度
        # print(len(data['far']))
        # print(len(data['thresholds']))
        pd_list.append(uniform_sample(data["pd"]))
        far_list.append(uniform_sample(data["far"]))
        if thresholds is None:
            thresholds = uniform_sample(data["thresholds"])

    return thresholds, pd_list, far_list
# 获取模拟数据
# tau, pd_list, far_list = generate_mock_data()
datapath = "Indian_pines_corrected"# Indian_pines_corrected
base_dir = f"/data1/wz/tsttd99916/csv/{datapath}.mat/"  # 改成你保存 npz 文件的目录
tau, pd_list, far_list = load_real_data_from_npz(base_dir, methods)


# Figure layout
fig = plt.figure(figsize=(18, 4))

# 图1: 3D ROC 曲面图
ax1 = fig.add_subplot(141, projection='3d')
for i, method in enumerate(methods):
    ax1.plot(far_list[i], pd_list[i], tau, label=method,
             color=colors[i], linestyle=linestyles[i])
ax1.set_xlabel('false alarm rate')
ax1.set_ylabel('probability of detection')
ax1.set_zlabel('τ')
# ax1.grid(False)
# ax1.legend()

# 图2: 对数 ROC 曲线
ax2 = fig.add_subplot(142)
for i, method in enumerate(methods):
    ax2.plot(far_list[i], pd_list[i], label=method,
                 color=colors[i], linestyle=linestyles[i])
ax2.set_xlabel('false alarm rate')
ax2.set_ylabel('probability of detection')
ax2.set_xlim(0, 0.2)  # 设置横坐标范围
ax2.set_ylim(0.8, 1)   # 设置纵坐标范围
ax2.legend()

# 图3: FAR vs τ
ax3 = fig.add_subplot(143)
for i, method in enumerate(methods):
    ax3.plot(tau, far_list[i], label=method,
             color=colors[i], linestyle=linestyles[i])
ax3.set_xlabel('τ')
ax3.set_ylabel('false alarm rate')
ax3.legend()

# 图4: PD vs τ
ax4 = fig.add_subplot(144)
for i, method in enumerate(methods):
    ax4.plot(tau, pd_list[i], label=method,
             color=colors[i], linestyle=linestyles[i])
ax4.set_xlabel('τ')
ax4.set_ylabel('probability of detection')
ax4.set_xlim(0.1, 0.9)  # 设置横坐标范围
ax4.set_ylim(0.1, 0.9)   # 设置纵坐标范围
ax4.legend()

plt.tight_layout()
plt.show()
save_dir = "picture/roc0422"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{datapath}.png")
plt.savefig(save_path)
print("save picture to {}".format(save_path))
