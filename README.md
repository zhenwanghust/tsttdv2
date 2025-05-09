# [高光谱遥感目标检测算法改进]

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

本仓库为华中科技大学《高级机器学习理论课程报告》对应的代码库。课题选择为第二个题目，即为：扩展算法解决竞赛问题或实际问题。

---

## 🚀 快速开始

### 环境要求
```text
torch==2.2.0
numpy==1.26.3
scipy==1.12.0
scikit-learn==1.4.0
matplotlib==3.8.2
tqdm==4.66.1
einops==0.7.0
mamba-ssm==1.1.1
```

### 安装部署
```bash
# 克隆仓库
git clone https://github.com/zhenwanghust/tsttdv2.git

# 进入项目目录
cd tsttdv2

# 安装依赖
pip install -r requirements.txt
```
---

## 🚀 测试与训练


### **验证预训练模型**
若要复现论文中报告的结果，请按以下步骤操作：
1. 在配置中设置：
   ```python
   state = "eval"                  # 运行模式设为评估
   band = "对应数据集光谱数"      # 根据数据集选择（如 189）
   path = "数据集路径"             # 数据存放路径（如 `Sandiego.mat`）
   name = "数据集名称（不含后缀）"  # 如 `Sandiego`
   test_load_weight = "finalbest"  # 已经提供了五个数据集训练好的权重
   ```  
2. 运行命令：
   ```bash
   python Main.py
   ```

### **复现训练过程**
若要重新训练模型并生成结果：
1. **训练阶段**
   ```python
   state = "train"                 # 运行模式设为训练
   band = "对应数据集光谱数"       # 同上
   path = "数据集路径"             # 同上
   name = "数据集名称（不含后缀）"  # 同上
   ```  
   运行命令：
   ```bash
   python Main.py
   ```

2. **评估阶段**
   训练完成后，修改配置：
   ```python
   state = "eval"  
   test_load_weight = "best_model_0_98_0_98_pt"  # 加载训练生成的权重
   ```  
   再次运行：
   ```bash
   python Main.py
   ```

---


## 🚀 额外信息

- 本课题的基准方法来自IEEE Trans. Geosci. Remote Sens.的文章[Triplet Spectralwise Transformer Network for Hyperspectral Target Detection](https://github.com/shendb2022/TSTTD)。该基准方法的代码实现由[Dubin Shen](https://github.com/shendb2022)完成。
- 基准方法可以在本实验的环境下运行
- 原论文信息如下
```
@ARTICLE{10223236,
  author={Jiao, Jinyue and Gong, Zhiqiang and Zhong, Ping},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Triplet Spectralwise Transformer Network for Hyperspectral Target Detection}, 
  year={2023},
  volume={61},
  number={},
  pages={1-17},
  keywords={Training;Transformers;Feature extraction;Hyperspectral imaging;Detectors;Object detection;Task analysis;Balanced learning;hyperspectral image;spectralwise transformer;target detection;triplet network},
  doi={10.1109/TGRS.2023.3306084}}
```
  
