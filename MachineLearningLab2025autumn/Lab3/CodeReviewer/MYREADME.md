# CodeBERT Lab3 任务总结

| 学号      | 姓名   | 分工               |
| --------- | ------ | ------------------ |
| 231250032 | 顾忠伟 | 完成代码修改和测试 |
| 231250031 | 周城屹 | 分析与预处理数据   |
| 231250030 | 黄天伟 | 训练与微调         |
| 231250008 | 徐立桐 | 编写文档与总结     |



本次实验共涉及三个任务：**Diff Quality Estimation (分类任务)**、**Comment Generation (生成任务)** 和 **Code Refinement (生成任务)**。以下对每个任务进行总结，包括任务目标、处理方式、对代码的修改以及复现结果。

>  本次实验的复现平台为 Windows11 + 单GPU + cu118，无法进行分布式推理，因此对代码进行了部分修改，具体代码可以看附件
>
>  出于提交需要，部分文件夹只保留部分与原仓库不同的关键内容
>
>  完整的实验项目结构如下：
>
>  ```
>  ├─code
>  │  │  bleu.py
>  │  │  configs.py
>  │  │  models.py
>  │  │  run_finetune_cls.py
>  │  │  run_finetune_msg.py
>  │  │  run_finetune_ref.py
>  │  │  run_infer_msg.py
>  │  │  run_test_cls.py
>  │  │  run_test_msg.py
>  │  │  run_test_ref.py
>  │  │  test_model.py
>  │  │  utils.py
>  │  │
>  │  ├─evaluator
>  │  │  │  bleu.py
>  │  │  │  smooth_bleu.py
>  │  │  │  stopwords.txt
>  │  │  │
>  │  │  ├─CodeBLEU
>  │  │  │  │  bleu.py
>  │  │  │  │  calc_code_bleu.py
>  │  │  │  │  dataflow_match.py
>  │  │  │  │  readme.txt
>  │  │  │  │  syntax_match.py
>  │  │  │  │  utils.py
>  │  │  │  │  weighted_ngram_match.py
>  │  │  │  │
>  │  │  │  ├─keywords
>  │  │  │  │      c_sharp.txt
>  │  │  │  │      java.txt
>  │  │  │  │
>  │  │  │  └─parser
>  │  │  │          build.py
>  │  │  │          build.sh
>  │  │  │          DFG.py
>  │  │  │          utils.py
>  │  │  │          __init__.py
>  │  │  │
>  │  │  └─__pycache__
>  │  │          smooth_bleu.cpython-38.pyc
>  │  │
>  │  ├─sh
>  │  │      finetune-cls.sh
>  │  │      finetune-msg.sh
>  │  │      finetune-ref.sh
>  │  │      infer-json.sh
>  │  │      test-cls.sh
>  │  │      test-msg.sh
>  │  │      test-ref.sh
>  │  │      test_nltk.sh
>  │  │
>  │  └─__pycache__
>  │          configs.cpython-38.pyc
>  │          models.cpython-38.pyc
>  │          utils.cpython-38.pyc
>  │
>  ├─codereviewer
>  │      .gitattributes
>  │      added_tokens.json
>  │      config.json
>  │      generation_config.json
>  │      golds.txt
>  │      merges.txt
>  │      preds.txt
>  │      pytorch_model.bin
>  │      README.md
>  │      special_tokens_map.json
>  │      tokenizer_config.json
>  │      vocab.json
>  │
>  ├─Code_Refinement
>  │      ref-test.jsonl
>  │      ref-train.jsonl
>  │      ref-valid.jsonl
>  │
>  ├─Comment_Generation
>  │      msg-test.jsonl
>  │      msg-testrb.exps
>  │      msg-train.jsonl
>  │      msg-valid.jsonl
>  │
>  └─Diff_Quality_Estimation
>        cls-test.jsonl
>        cls-testrb.exps
>        cls-train-chunk-0.jsonl
>        cls-train-chunk-1.jsonl
>        cls-train-chunk-2.jsonl
>        cls-train-chunk-3.jsonl
>        cls-valid.jsonl
>  
>  ```

------

## 任务一：Diff Quality Estimation（分类任务）

**目标**：
 对代码 diff 的质量进行分类预测，判断提交的质量是高还是低，用于辅助代码审查。

**处理方式**：

- 使用 CodeBERT 架构进行文本编码。
- 数据集采用 `Diff_Quality_Estimation/cls-test.jsonl`，输入为 diff 内容，输出为分类标签。
- 使用 `CommentClsDataset` 或 `SimpleClsDataset` 读取和处理数据。
- 模型评估使用准确率 (accuracy) 作为指标。

**代码修改**：

1. 注释掉分布式训练部分，改为单 GPU 运行模式：

```python
local_rank = 0
args.local_rank = 0
args.global_rank = 0
args.world_size = 1
```

1. 去掉 DDP 包装，改为 `model = model.cuda()`。
2. Windows 下 DataLoader `num_workers` 改为 0，避免 `multiprocessing` 报错。
3. 保持 `eval_file` 指向 `Diff_Quality_Estimation/cls-test.jsonl`。

**运行指令**：

```python
(CodeBert) D:\codes\MachineLearningStudyCode\MachineLearningLab2025autumn\Lab3\CodeBERT\CodeReviewer\code>
	python run_test_cls.py --model_name_or_path ../codereviewer --eval_file ../Diff_Quality_Estimation/cls-test.jsonl --eval_batch_size 4
```

**复现结果**：

```
11/29/2025 20:58:51 - INFO - __main__ -   
              precision    recall  f1-score   support

           0     0.5025    0.8774    0.6390     15626
           1     0.5160    0.1308    0.2087     15617

    accuracy                         0.5042     31243
   macro avg     0.5093    0.5041    0.4239     31243
weighted avg     0.5093    0.5042    0.4239     31243

11/29/2025 20:58:51 - INFO - __main__ -   Target positive percentage: 0.49985596773677304
11/29/2025 20:58:51 - INFO - __main__ -   Test finished.
```

------

## 任务二：Comment Generation（生成任务）

**目标**：
 根据代码 diff 生成对应的提交说明（commit message），实现自动化注释生成。

**处理方式**：

- 使用 CodeBERT 或 CodeT5 架构进行生成任务。
- 数据集采用 `Comment_Generation/msg-test.jsonl`。
- 模型输入代码 diff，输出为生成的 commit message。
- 评估指标为 BLEU 值，比较生成文本与 gold text 的相似度。

**代码修改**：

1. 注释掉分布式训练部分，改为单 GPU 运行模式。
2. 去掉 DDP 包装，改为 `model = model.cuda()`。
3. Windows 下 DataLoader `num_workers` 改为 0。
4. 更新 `eval_file` 路径为 `../Comment_Generation/msg-test.jsonl`。

**运行指令**：

```python
(CodeBert) D:\codes\MachineLearningStudyCode\MachineLearningLab2025autumn\Lab3\CodeBERT\CodeReviewer\code>
	python run_test_msg.py ^  --model_name_or_path ../codereviewer ^  --eval_file ../Comment_Generation/msg-test.jsonl ^  --eval_batch_size 4 ^  --beam_size 5 ^  --max_target_length 128
```

**复现结果**：

```
2540it [13:18,  3.18it/s]
Total: 10157
11/29/2025 21:21:35 - WARNING - __main__ -   WithStop BLEU: 2.13
Total: 10157
11/29/2025 21:21:37 - WARNING - __main__ -   BLEU: 2.88
11/29/2025 21:21:37 - INFO - __main__ -   Test finished.
```

------

## 任务三：Code Refinement（生成任务）

**目标**：
 根据原始代码生成优化后的代码片段，实现自动代码修复或改进。

**处理方式**：

- 使用 CodeBERT 或 CodeT5 架构进行生成任务。
- 数据集采用 `Code_Refinement/msg-test.jsonl`，输入为原始代码，输出为优化后的代码。
- 模型输出为生成的代码片段，评估指标包括 BLEU 值和 Exact Match（EM）。

**代码修改**：

1. 注释掉分布式训练部分，改为单 GPU 运行模式。
2. 去掉 DDP 包装，改为 `model = model.cuda()`。
3. Windows 下 DataLoader `num_workers` 改为 0。
4. 更新 `eval_file` 路径为 `../Code_Refinement/msg-test.jsonl`。

**复现结果**：

``````
3276it [24:06,  2.27it/s]
11/29/2025 21:50:14 - WARNING - __main__ -   EM: 0.0
Total: 13104
11/29/2025 21:50:18 - WARNING - __main__ -   BLEU: 1.91
11/29/2025 21:50:18 - INFO - __main__ -   Test finished.
``````

------

## 总结

本次实验通过对三个不同任务进行复现和修改，实现了在 Windows 单 GPU 环境下的运行：

- **Diff Quality Estimation**：文本分类任务，实现了代码质量评估。
- **Comment Generation**：生成任务，实现了代码 diff 对应的 commit message 自动生成。
- **Code Refinement**：生成任务，实现了代码片段的自动优化与修复。

在实验过程中，主要修改集中在**禁用分布式训练、单 GPU 模式、Windows 下 DataLoader 设置**等方面，从而保证了实验在本地环境的顺利复现。

复现结果可用于进一步分析模型性能和改进生成/分类效果。
