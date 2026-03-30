# Machine Learning Study Code

机器学习课程实验代码仓库，包含 Lab1-Lab6 多种机器学习任务。

## 目录结构

```
MachineLearningStudyCode/
├── MachineLearningLab2025autumn/    # 主要实验代码
│   ├── Lab1/        # PR 处理时间预测 & 是否合入分类（传统机器学习）
│   ├── Lab2/        # 多项目 PR 预测（PyTorch MLP 神经网络）
│   ├── Lab3/        # CodeReviewer 代码审查预训练模型
│   ├── Lab4/        # GAN 生成对抗网络（Basic GAN, DCGAN, WGAN-GP）
│   ├── Lab5/        # NLP 任务（逻辑回归、朴素贝叶斯、词向量、LSH 近似检索）
│   ├── Lab6/        # LLM 微调（Prompt Engineering、PEFT、PPO 对齐）
│   └── requirements.txt
└── test/             # 基础机器学习示例（糖尿病预测、鸢尾花、MNIST 等）
```

## 环境配置

```bash
cd MachineLearningLab2025autumn
pip install -r requirements.txt
```

主要依赖：PyTorch 2.4.1+cu118、scikit-learn、pandas、numpy、matplotlib

## 运行实验

所有 Lab 的执行以 `MachineLearningLab2025autumn/` 为根目录：

```bash
cd MachineLearningLab2025autumn

# Lab1
python Lab1/main.py

# Lab2
python Lab2/task1.py    # PR 处理时间回归
python Lab2/task2.py    # PR 合入分类

# Lab5 Task4（LSH 相似推文检索）
cd Lab5/Task4
python translate.py
```

## 各 Lab 简介

| Lab | 主题 | 技术栈 |
|-----|------|--------|
| Lab1 | PR 预测（单项目） | 线性回归、岭回归、逻辑回归、随机森林 |
| Lab2 | PR 预测（多项目） | PyTorch MLP 神经网络 |
| Lab3 | 代码审查 | CodeReviewer 预训练模型、Transformer |
| Lab4 | 图像生成 | GAN、DCGAN、WGAN-GP（PyTorch） |
| Lab5 | NLP 任务 | 词向量、朴素贝叶斯、LSH |
| Lab6 | LLM 对齐 | Prompt Engineering、PEFT/LoRA、PPO |

## Lab5 Task4 额外依赖

LSH 任务需要下载 NLTK 数据：

```bash
python -c "import nltk; nltk.download('twitter_samples'); nltk.download('stopwords')"
```
