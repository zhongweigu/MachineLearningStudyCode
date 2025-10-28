# 小组信息

顾忠伟 231250032 : 模型训练部分代码编写
周城屹 231250031 : 数据集收集与预处理
黄天伟 231250030 : 设计方案和文档
徐立桐 231250008 : 执行训练过程与微调

---

# 实验报告

## 实验内容

本实验为《实验作业2》，完成 **Level 1**：

* 在实验1的基础上，将分析范围扩展到 **多个项目（共10个开源项目）**。
* 完成两个任务：

  * 任务一：使用神经网络预测 PR 处理时间（TTC）。
  * 任务二：使用神经网络预测 PR 是否被合并（is_merged）。
* 本实验重点在于：

  * 多项目数据融合与时间切分；
  * 使用 **PyTorch** 实现多层感知机（MLP）神经网络；
  * 输出回归与分类性能指标。

---

## 一、问题与数据

### 任务一：预测 Pull Request 处理时间

* **定义**：以 `TTC (Time-to-Close)` 为目标变量，表示从 PR 创建到关闭/合并的时间（小时）。
* **目标**：使用神经网络回归模型预测 PR 处理时长。

### 任务二：预测 Pull Request 是否合入

* **定义**：判断 PR 是被合并 (merged) 还是关闭但未合并 (closed)。
* **目标**：使用神经网络分类模型完成二分类预测。

---

### 数据来源

* 数据文件：`Lab2/documents/<project>/processed_features.xlsx`
* 每个项目对应一个 `processed_features.xlsx` 文件，包含 Lab1 中提取的特征。
* 共计 10 个项目数据。

---

### 时间切分策略

为保证时间上的因果性和泛化性，采用固定时间划分：

* 训练集：PR 创建时间 < 2021-06-01
* 测试集：2021-06-01 ≤ 创建时间 ≤ 2022-06-15

---

## 二、特征工程

* **时间字段**：`created_at`, `updated_at`, `closed_at`, `merged_at` 均转换为 `datetime` 格式。

* **衍生特征：**

  * `TFR` (Time-to-First-Response)：从创建到首次更新（小时）
  * `TTC` (Time-to-Close)：从创建到关闭/合并（小时）
  * `is_merged`：是否被合并 (1=合并, 0=未合并)

* **特征处理：**

  * 删除 `modify_proportion`, `modify_entropy` 等无效列；
  * 缺失值统一填充为 0；
  * 将 `inf/-inf` 替换为 0；
  * 数值特征标准化（StandardScaler）；
  * 非数值特征（时间、编号、标签等）不参与训练。

* **多项目整合：**

  * 从 `Lab2/documents` 目录中读取各项目的 processed_features；
  * 添加 `project` 列用于标识来源；
  * 按行合并成统一训练集。

---

## 三、模型与方法

### （一）任务一：PR 处理时间回归

**模型结构（MLPRegressor）：**

| 层级      | 维度              | 激活函数 |
| ------- | --------------- | ---- |
| Linear  | input_dim → 256 | ReLU |
| Dropout | 0.3             | —    |
| Linear  | 256 → 128       | ReLU |
| Dropout | 0.2             | —    |
| Linear  | 128 → 1         | —    |

**优化与训练参数：**

* Optimizer：Adam (lr=1e-3, weight_decay=1e-5)
* Loss：MSELoss
* Epochs：50
* Batch size：64
* 设备：自动检测 GPU (cuda) 或 CPU

---

### （二）任务二：PR 合入分类

**模型结构（MLPClassifier）：**

| 层级      | 维度              | 激活函数    |
| ------- | --------------- | ------- |
| Linear  | input_dim → 256 | ReLU    |
| Dropout | 0.3             | —       |
| Linear  | 256 → 128       | ReLU    |
| Dropout | 0.2             | —       |
| Linear  | 128 → 1         | Sigmoid |

**优化与训练参数：**

* Optimizer：Adam (lr=1e-3)
* Loss：BCELoss（二分类交叉熵）
* Epochs：50
* Batch size：64
* 设备：cuda / cpu 自动选择

---

## 四、结果与分析

### 任务一：PR 处理时间预测（回归）

* 训练集大小: (167300, 1189)
* 测试集大小: (14189, 1189)
* 特征维度: 1181
* 设备：GPU（cuda）

| 指标   | 值            |
| ---- | ------------ |
| MAE  | 711.66       |
| MSE  | 3,622,902.96 |
| RMSE | 1903.39      |
| R²   | -0.1630      |

**分析：**

* 回归结果整体较弱（R² 为负），说明模型未能准确拟合处理时间。
* 原因推测：

  * PR 处理时间波动大、受项目管理策略影响明显；
  * 不同项目间时间尺度差异较大；
  * 神经网络虽捕捉到一定规律，但偏差仍大。

---

### 任务二：PR 合入预测（分类）

* 训练集大小: (167300, 1183)
* 测试集大小: (14189, 1183)
* 特征维度: 1183

| 指标        | 值      |
| --------- | ------ |
| Accuracy  | 0.8003 |
| Precision | 0.8283 |
| Recall    | 0.9251 |
| F1        | 0.8740 |

**分析：**

* 模型表现良好，能有效区分被合并与未合并的 PR。
* 召回率高（≈0.93），说明模型几乎能识别出所有“会被合并”的 PR。
* 精确率与 F1 较高，表明模型泛化能力良好。
* 对比 Lab1 的传统模型（F1≈0.68），神经网络在多项目融合后显著提升分类性能。

---

## 五、结论与建议

* **任务一**：神经网络在多项目数据下仍难以准确预测具体处理时长，说明该问题存在较强的外部随机性。
* **任务二**：神经网络分类器表现稳定优异，说明多项目融合后模型能学习到跨项目的通用合并模式。
* **改进方向：**

  * 引入更多上下文特征（如代码复杂度、开发者社交网络等）；
  * 使用时间衰减权重、项目嵌入等提高跨项目适配性；
  * 引入验证集 + early stopping 优化模型稳定性。

---

# 补充说明

* 处理后的多项目数据位于：
  `Lab2/documents/<project>/processed_features.xlsx`
* 运行环境：

  * Python: 3.11.7
  * PyTorch: 2.x
  * 依赖库：详见 `requirements.txt`
* 运行路径建议：

  * 项目根目录为 `MachineLearningLab2025autumn`
  * 运行脚本时路径形如：
    `python Lab2/task1.py`
    `python Lab2/task2.py`

---

# 项目结构

```
│  requirements.txt
│
├─Lab1
│   │  main.py
│   │  train_task1.py
│   │  train_task2.py
│   │  实验作业1.pdf
│   │  README.md
│   └─yii2
│       ├─ author_features.xlsx
│       ├─ processed_features.xlsx
│       ├─ project_features.xlsx
│       ├─ ...
│
└─Lab2
    │  main.py
    │  task1.py     # 神经网络回归任务（预测TTC）
    │  task2.py     # 神经网络分类任务（预测是否合入）
    │  实验作业2.pdf
    │  README.md
    └─documents
        ├─ <project1>/processed_features.xlsx
        ├─ <project2>/processed_features.xlsx
        ├─ ...
```
