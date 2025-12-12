# 实验报告：任务一 —— Prompt Engineering 基础实验

文件依据：Prompt-Engineering-Basics.html（已解析）
实验主题：对话摘要任务中的 Zero-shot / One-shot / Few-shot Prompt Engineering

---

## 1. 实验背景

本实验使用 Hugging Face 的 **FLAN-T5-Base** 模型完成 DialogSum 数据集的对话摘要任务，并在三个场景下观察模型行为：

1. **Zero-shot**：无指令、无示例
2. **One-shot**：提供一条完整示例
3. **Few-shot（Exercise）**：提供多条示例，修改示例数量与索引

每个 Exercise 均要求：

* 按 Notebook 要求修改参数
* 重现输出
* 分析模型行为原因（insight）
* 回答 Notebook 中提出的问题

以下逐项展开。

---

# 2. Exercise 1：Zero-shot 结果分析

### 2.1 操作步骤

Notebook 中直接运行 Zero-shot 推理，无需构造任何 prompt，只将一段对话输入模型：

模型输出示例（来自文件）：

```
MODEL GENERATION - WITHOUT PROMPT ENGINEERING:
#Person1#: I'm thinking of upgrading my computer.
```



### 2.2 观察到的现象

* 模型 **没有生成摘要**
* 它 **续写了对话**（继续生成下一句）
* 输出内容往往与真实摘要完全不一致

例如：
真实摘要（baseline human summary）：

```
#Person1# teaches #Person2# how to upgrade software and hardware...
```

模型输出却是：

```
#Person1#: I'm thinking of upgrading my computer.
```

### 2.3 原因分析（insight）

Zero-shot 情况下，模型缺乏“任务指令”与“格式示例”，因此它会：

* 默认按照训练时最常见的任务模式——**对话续写（next-token prediction）**
* 并不知道“你希望它做摘要”
* 因此输出行为偏离任务本意

**核心 insight：
在无 prompt 指令的情况下，大模型不会自动推断出“我要做摘要”，而是回到语言模型的默认行为：继续写下去。**

### 2.4 本 Exercise 的结论

Zero-shot 对话摘要效果非常差，主要原因不是模型能力不足，而是 **prompt 中缺乏任务定义**。这为 prompt engineering 的必要性奠定基础。

---

# 3. Exercise 2：One-Shot Inference（任务 3.1）

### 3.1 操作步骤

Notebook 构造 one-shot prompt：

* 选择一个示例：`example_indices_full = [40]`
* 设置目标样例：`example_index_to_summarize = 200`

Prompt 内容如文件所示：

```
Dialogue:
(完整示例对话 40)

What was going on?
(示例摘要)

Dialogue:
(目标对话 200)

What was going on?
```



运行模型后的输出：

```
MODEL GENERATION - ONE SHOT:
#Person1 wants to upgrade his system. #Person2 wants to add a painting program...
```



### 3.2 观察到的现象

与 Zero-shot 相比：

* 模型开始**生成摘要**而不是续写
* 摘要结构和语言风格更贴近真实标签
* 但仍存在细节偏差 / 简化过度

### 3.3 原因分析（insight）

One-shot 的优势来自 **In-Context Learning**：

1. 给出一个完整的“输入 → 输出”示例
2. 模型自动识别“模式”
3. 模型迁移示例结构到下一段对话

但由于只提供一个示例：

* 模型掌握的任务分布有限
* 仍容易遗漏细节或做过度抽象总结

### 3.4 小结

One-shot 明显好于 Zero-shot，但精确性受限于单示例的多样性不足。

---

# 4. Exercise 3：Few-Shot Inference（任务 4.2）

### 4.1 Notebook 要求

```
Experiment with the few shot inferencing.
- Choose different dialogues by modifying example_indices_full and example_index_to_summarize.
- Change the number of shots, but stay within 512 context length.
How well does few shot inferencing work with other examples?
```



### 4.2 操作步骤

Notebook 默认选择：

```
example_indices_full = [40, 80, 120]
example_index_to_summarize = 200
```

生成 few-shot prompt → 模型输出：

```
MODEL GENERATION - FEW SHOT:
#Person1 offers some suggestions about how to upgrade her system.
```



### 4.3 观察到的现象

与 One-shot 相比：

* 输出依然是摘要
* 质量略有改善，但 **改进幅度有限**
* 某些情况下甚至与 One-shot 表现相似（甚至略退步）

文件中明确指出：

```
In this case, few shot did not provide much of an improvement over one shot inference.
Anything above 5 or 6 shots typically not help much.
```



### 4.4 原因分析（insight）

Few-shot 理论上应提供更多任务模式信息，但效果有限的原因是：

1. **模型输入长度限制（512 tokens）**

   * 超过后模型会截断重要示例
   * 影响学习质量

2. **示例之间的任务分布差异较大**

   * 三个对话情节差异大，模型难以抽取稳定模式
   * 反而导致泛化不佳

3. **FLAN-T5 已经经过 instruction finetune**

   * 对“对话摘要”任务可能已有能力
   * 更多示例反而边际收益递减

4. **Prompt 结构较长且重复，对齐难度提升**

   * 模型可能只学习了格式，而非深层语义

**核心 insight：
Few-shot 不是“示例越多越好”，关键是“代表性强、结构一致、数量适中”。在本实验中，由于示例差异大 + 输入长度限制，few-shot 增益被抵消了。**

### 4.5 回答 Exercise 的问题

**“How well does few shot inferencing work with other examples?”**

基于实验结果与 Notebook 输出：

* Few-shot 对摘要质量有 **轻微提升，但不稳定**
* 某些例子中几乎与 One-shot 相当
* 超过 3~5 个示例通常不会带来更多收益
* 若示例风格差异大，模型甚至可能退化
* 受 FLAN-T5 的 512 token context length 限制，过多示例会被截断，直接损害性能

总结：
**Few-shot 在本任务中效果有限，不一定优于 One-shot，示例数量过大或不一致反而会降低质量。**

---

# 5. 综合总结

本任务的核心认识：

1. **Zero-shot → 续写而非摘要**

   * 原因：缺乏任务指令，模型回到语言模型默认行为。

2. **One-shot → 能正确生成摘要**

   * 模型通过示例进行 In-Context Learning，掌握“对话 → 摘要”模式。

3. **Few-shot → 改善有限且不稳定**

   * 因示例不一致、上下文长度限制导致边际收益下降
   * 超过 5~6 shots 一般无益

整体 insight：
**Prompt Engineering 的效果高度依赖示例质量与上下文长度，而非单纯增加示例数量。**

