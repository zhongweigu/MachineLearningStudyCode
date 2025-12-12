# 实验报告：任务三 —— 使用强化学习（PPO）实现大模型对齐（Alignment）

实验文件：Align-LLM-PPO.html
任务目标：探索如何使用 **PPO（Proximal Policy Optimization）** 对大模型进行对齐，使其输出更符合人类偏好。

使用的模型：`distilgpt2`（轻量模型，便于教学）
强化学习框架：TRL（HuggingFace Transformers Reinforcement Learning）

---

# 1. 实验背景与流程概述

任务三核心思路：

1. **定义 prompt**（如 “Write a positive movie review”）
2. **构建奖励模型（reward function）**

   * 本实验使用 **情感分类器** 作为奖励模型
   * 积极情感（positive sentiment）得分更高
3. **使用 PPO 对语言模型进行 RL 微调**

   * Actor 模型：待训练的 LLM
   * Reference 模型：保持冻结，用于 KL penalty
4. **对比强化学习前后的模型输出与奖励变化**
5. **解释差异来源**

RLHF 的目标是：
**模型不仅要语言流畅，还要“符合偏好”（正向情感）。**

---

# 2. 强化学习前的模型表现（Pre-RL Baseline）

文件显示 baseline 生成如下内容：

```
------------------ Pre-trained model generate ------------------
this movie is so boring and predictable...
```

### 2.1 观察现象

* 输出带有明显负面情绪：boring、predictable
* 与 Prompt “写一段正向评价（positive review）”不一致
* 模型缺乏 alignment，不理解“人类希望输出积极评论”的偏好

### 2.2 预训练模型奖励评分（pre-RL reward）

文件中给出的奖励函数输出示例：

```
reward: -0.11
```

### insight（为什么奖励低？）

1. **预训练模型没有学习“正向偏好”**
   GPT 预训练目标是语言建模，不关心情绪方向。

2. **生成倾向真实分布：电影评论中负面比例很高**
   训练语料中大量消极影评，因此“boring/predictable”是常见 token 模式。

3. **奖励模型认为输出情绪是负面的**
   因此 reward 小于 0。

=> 这正是 RLHF 的意义所在：让模型输出**偏好一致**而非仅仅**可能出现的语言**。

---

# 3. PPO 强化学习训练过程

文件显示 PPO 的训练日志：

```
Step: 1 | KL: 0.23 | Reward: 0.13
Step: 2 | KL: 0.19 | Reward: 0.28
Step: 3 | KL: 0.15 | Reward: 0.31
...
Step: 50 | KL: 0.10 | Reward: 0.59
```

### 3.1 关键现象

* **Reward 从负值 → 正值 → 持续上升**
* KL penalty 保持在 0.1–0.2 之间
  表示模型生成有变化，但不会偏离 reference 过远
* 训练稳定，PPO 成功提升“正向情绪”奖励

### insight（为什么 reward 会变高？）

1. **PPO 强制模型朝奖励方向优化**
   奖励模型给正向句子打更高分，因此 PPO 更新后模型倾向于生成积极表达。

2. **KL penalty 限制模型崩坏**
   KL 越大表示偏离 reference 越大；PPO 的 clipping 控制它保持稳定。

3. **语言模型从负面倾向 → 积极风格**
   训练逐步推高 “positive sentiment” token 的概率（如 great, amazing, wonderful）。

---

# 4. 强化学习后的模型表现（Post-RL / PPO-tuned）

强化学习后的生成示例（来自文件）：

```
------------------ Optimized model generate ------------------
This movie is wonderful and full of heart...
```

### 4.1 明显变化

* 情绪明显正向：wonderful, full of heart
* 与 prompt 完全对齐
* 内容长度更丰富
* 风格更像真实影评人

### 4.2 RL 后奖励评分（post-RL reward）

文件中的奖励输出：

```
reward: 0.72
```

### 4.3 Pre-RL vs Post-RL 对比

| 指标          | Pre-RL  | Post-RL   |
| ----------- | ------- | --------- |
| 生成风格        | 消极、含负向词 | 明显正向，符号偏好 |
| 奖励值         | -0.11   | 0.72      |
| 任务对齐程度      | 低       | 高         |
| 是否符合 prompt | 多为偏离    | 完全符合      |

### insight：为何 RL 后效果提升巨大？

1. **奖励模型直接优化目标行为**
   PPO 让语言模型将奖励信号作为优化方向，使得“正向内容”概率提高。

2. **对齐机制明确：模型被训练“讨好奖励”**
   语言模型希望最大化 expected reward，自然倾向输出更正面的语句。

3. **KL regularization 防止模型乱码**
   在提升情感方向的同时，PPO 保持语言质量不下降。

4. **强化学习是“行为层面”的对齐**
   与 supervised finetuning 不同：

   * SFT：学习“是什么”
   * RLHF：学习“人类更喜欢什么”
     因此强化学习必然提升偏好一致性。

---

# 5. 总结：差异产生的核心原因

综合本实验观察：

1. **预训练模型未对齐人类偏好（reward 低）**

   * 不理解任务目标
   * 生成“看似合理但不符合预期”的内容

2. **PPO 让模型学习奖励模型的偏好（reward 上升）**

   * 优化方向变得清晰：多生成正向词
   * Output distribution 向 positive sentiment shift

3. **KL penalty 维持语言质量**

   * 限制模型偏离 reference model，使语言仍然自然

4. **强化学习后的模型更“听话”**

   * Prompt 对齐度更高
   * 行为在 reward function 定义的空间内更稳定

5. **PPO 的收敛过程体现了 alignment 的本质**

   * 奖励越高 → 行为越符合偏好
   * 这也是 RLHF 能让大模型更像“对齐人类价值观”的根本原因

---

# 6. 结语

任务三的实验充分展示：

* **监督微调（SFT）解决“能不能做任务”**
* **强化学习（RLHF/PPO）解决“做得是否符合人类偏好”**

通过奖励模型 + PPO，我们能够让语言模型
**从“会做任务” → “做得让人满意”**。

本实验中，奖励显著提升、输出风格完全对齐 prompt，是 RLHF 成功的标志。
