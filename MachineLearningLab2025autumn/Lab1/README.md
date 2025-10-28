# 小组信息

顾忠伟 231250032

周城屹 231250031

黄天伟 231250030

徐立桐 231250008


# 实验报告

## 实验内容

本实验为《实验作业1》，完成 **Level 1 基础层**：

* 使用课程提供的单一项目数据（yii2 仓库 PR 数据），进行 **预测 PR 处理时间** 与 **预测 PR 是否合入** 两个任务。
* 实验未涉及真实数据爬取与多项目扩展。

---

## 一、问题与数据

* **任务一：预测 Pull Request 处理时间**

  * 定义：以 `TTC (Time-to-Close)` 为目标变量，表示从 PR 创建到关闭/合并的时间（小时）。
  * 目标：使用回归模型预测 PR 处理时间。

* **任务二：预测 Pull Request 是否合入**

  * 定义：判断 PR 是被合并 (merged) 还是关闭但未合并 (closed)。
  * 目标：使用分类模型完成二分类预测。

* **数据来源**：课程提供的 `PR_features.xlsx` 和 `pr_time_dict.txt`

* **时间切分**：

  * 训练集：2021-05-31 之前创建的 PR
  * 测试集：2021-06-01 及之后的 PR
  * 保证时间上的前后顺序，避免数据泄漏。

---

## 二、特征工程

* **原始特征**（来自 `PR_features.xlsx`）：

  * 关键词特征：是否包含 `bug/document/feature/improve/refactor/test` 等
  * 修改规模特征：`lines_added`，`lines_deleted`，`files_changed` 等
  * 作者特征：是否核心成员，历史 PR 数量等
  * 结构特征：目录数、文件类型数、语言类型数
  * 文本嵌入：PR 标题、正文、评论的 embedding 向量

* **时间特征**（由 `pr_time_dict.txt` 生成）：

  * `created_at`：PR 创建时间
  * `closed_at`：关闭时间
  * `merged_at`：合并时间

* **衍生标签**：

  * 任务一：计算 `TTC`（小时）
  * 任务二：生成 `is_merged`（1=合并，0=未合并）

* **处理方式**：

  * 缺失值填充为 0
  * 异常值（inf）替换为 0
  * 将高维向量元素拆分（如body_embedding拆成body_embedding_vec01,body_embedding_vec02...）
  * 数值特征标准化（StandardScaler）
  * 时间字段用于切分训练集与测试集，不作为模型输入

---

## 三、模型与方法

* **任务一：回归模型**

  * 线性回归（Linear Regression）
  * 岭回归（Ridge Regression）

* **任务二：分类模型**

  * 逻辑回归（Logistic Regression）
  * 随机森林（Random Forest Classifier）

* **训练过程**

  * 特征矩阵：去掉 ID、时间戳、标签列
  * 使用训练集拟合模型，在测试集上评估性能

---

## 四、结果与分析

### 任务一：PR 处理时间预测

* 训练集大小: (7361, 1188)
* 测试集大小: (605, 1188)
* 特征维度: 1181

| 模型               | MAE     | RMSE       | R²    |
| ---------------- | ------- | ---------- | ----- |
| LinearRegression | 1679.09 | 5427492.89 | -4.63 |
| Ridge            | 1672.83 | 5382315.93 | -4.58 |

**分析**：

* 结果非常差，R² 为负数，说明模型预测效果比简单的均值预测还差。
* 可能原因：

  * 数据规模有限（只有单项目，样本不足以捕捉规律）。
  * PR 处理时间受外部因素影响较大（如开发者活跃度、项目管理方式），仅依靠课程提供的少量特征难以建模。

---

### 任务二：PR 合入预测

* 训练集大小: (7361, 1183)
* 测试集大小: (605, 1183)

| 模型                 | Accuracy | Precision | Recall | F1     |
| ------------------ | -------- | --------- | ------ | ------ |
| LogisticRegression | 0.7752   | 0.6336    | 0.6639 | 0.6443 |
| RandomForest       | 0.7686   | 0.6655    | 0.7487 | 0.6810 |

**分析**：

* 分类任务表现较好，尤其是随机森林，能够捕捉到较复杂的非线性特征关系。
* Logistic Regression 在精确度和召回率上略低，但整体可接受。

---

## 五、结论与建议

* **任务一**：由于数据集规模有限且特征不足，预测 PR 处理时间的效果很差，说明该问题需要更多样本与更丰富的上下文特征。
* **任务二**：分类模型可以在一定程度上预测 PR 是否合入，随机森林表现优于逻辑回归，F1≈0.68，说明特征对 PR 合入结果有一定预测能力。
* **改进方向**：

  * 获取更多项目数据，提升样本量
  * 增加文本表示（如使用 BERT embedding）

# 补充说明

处理后的数据位于Lab1/yii2/processed_features.xlsx

项目使用python:3.11.7

环境依赖: 见requirements.txt

由于项目读取数据的路径是 Lab1/yii2/xxxx , 因此运行时项目根目录最好是 Lab1 的上一级文件夹

项目结构
```
│  README.md
│  requirements.txt
│
└─Lab1
    │  main.py
    │  train_task1.py
    │  train_task2.py
    │  实验作业1 .pdf
    │
    └─yii2
            author_features.xlsx
            processed_features.xlsx (运行main后生成)
            project_features.xlsx
            PR_comment_info.xlsx
            PR_commit_info.xlsx
            PR_features.xlsx
            PR_info.xlsx
            PR_info_add_conversation.xlsx
            pr_time_dict.txt
            reviewer_features.xlsx
```