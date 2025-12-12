# Task4：基于 LSH 的相似推文检索

本任务基于 `twitter_samples` 数据集，将推文映射到嵌入空间后使用 **随机超平面哈希（LSH: Locality-Sensitive Hashing）** 实现近似最近邻检索（Approximate KNN），用于高效查找语义相似的推文。

运行入口：`translate.py`（脚本参数与流程均在代码中固定）。

---

## 1. 整体流程概述

系统通过如下步骤完成相似检索：

1. **推文预处理与向量化**
   使用 `utils.process_tweet` 对推文进行清洗、分词、去噪。对每个标记词查找其词向量，并对推文所有词向量求和，形成 **Bag-of-Embeddings 文档向量**。

2. **LSH 建表（多宇宙结构）**
   生成 `N_UNIVERSES` 组超平面，每组包含 `N_PLANES` 条随机向量。
   每个超平面集合代表一个“宇宙”，对所有文档向量分别进行符号哈希，得到哈希桶并写入对应宇宙的 `hash_table` 与 `id_table`。

3. **近似 KNN 查询**
   对目标文档向量 `v`，在多个宇宙中查找其哈希桶内的文档 ID，合并并去重候选；随后对候选集中所有文档计算余弦相似度（使用 `utils.cosine_similarity`），排序后返回前 K 个结果。

该方案同时具备**较高查询速度**与**可接受的召回质量**，适合高维向量的近似检索场景。

---

## 2. 实现原理与关键细节

### 2.1 文档向量构建（Document Embedding）

文件 `en_embeddings.p` 提供词向量子集。
`get_document_embedding(tweet, en_embeddings)` 执行：

* 调用 `process_tweet` 生成标准化 token 序列。
* 将所有出现在 `en_embeddings` 中的词向量求和：

  ```
  doc_vec = Σ embedding(word)
  ```
* 若推文全为未知词，返回全零向量（同维度）。

`get_document_vecs(...)` 会对所有推文运行上述过程，返回：

* `document_vecs`: shape = `(M, D)` 的矩阵
* `ind2Tweet`: 文档行 index 与原始推文的映射

### 2.2 随机超平面哈希（LSH）

每个宇宙使用一组大小为 `(D, N_PLANES)` 的随机矩阵 `planes`。

**哈希方式：**
对文档向量 `v`：

1. 计算投影：`projections = v^T · planes`
2. 取符号位：`bits = projections >= 0`
3. 将 bit 串解释为二进制整数，得到哈希值 `h`

该整数即文档落入的哈希桶。

**建表：**
每个宇宙维护两个字典：

* `hash_table[h] = [向量列表]`
* `id_table[h] = [文档 ID 列表]`

### 2.3 多宇宙近似查询（Approximate KNN）

`approximate_knn(...)` 执行步骤：

1. **在多个宇宙中查询同哈希值桶，合并候选集**
2. **去重并移除原文档自身**
3. **对候选计算余弦相似度**
4. **按相似度降序排序，取 Top-K**

若目标文档在所用宇宙中未命中任何桶，则返回空列表。

---

## 3. 核心函数说明

| 函数                                                       | 功能                      |
| -------------------------------------------------------- | ----------------------- |
| `load_embeddings(path)`                                  | 读取词向量文件并规范化为 `float32`。 |
| `get_document_embedding(tweet, embeddings)`              | 将单条推文映射为文档向量。           |
| `get_document_vecs(all_docs, embeddings)`                | 批量构建文档矩阵和索引映射。          |
| `hash_value_of_vector(v, planes)`                        | 计算单个宇宙内的哈希值。            |
| `make_hash_table(vecs, planes)`                          | 构建该宇宙的哈希表与 ID 表。        |
| `approximate_knn(doc_id, v, planes_l, k, num_universes)` | 多宇宙查询 + 余弦排序，实现近似 KNN。  |
| `main()`                                                 | 完成数据加载、向量化、建表与演示查询。     |

---

## 4. 复现步骤

### 4.1 安装与数据准备

首次运行前，可手动下载 NLTK 数据：

```bash
python -c "import nltk; nltk.download('twitter_samples'); nltk.download('stopwords')"
```

脚本自身也会尝试优先从项目目录 `tmp2/` 读取本地 NLTK 资源。

确保已存在文件：

* `en_embeddings.p`（随作业提供）
* `utils.py`（包含 `process_tweet`, `cosine_similarity`）

### 4.2 运行脚本（Windows）

```bash
cd d:\codes\MachineLearningStudyCode\MachineLearningLab2025autumn\Lab5\Task4
python translate.py
```

脚本运行后会打印：

* 文档数量与向量矩阵形状
* 示例随机向量的哈希值
* 各宇宙哈希表构建进度
* 查询文档及其近似邻居的文本内容

---

## 5. 参数与可调性

| 参数名                    | 默认值 | 含义                  |
| ---------------------- | --- | ------------------- |
| `N_PLANES`             | 10  | 每个宇宙的超平面数量，越大哈希越稀疏。 |
| `N_UNIVERSES`          | 25  | 设置多少个互相独立的 LSH 空间。  |
| `num_universes_to_use` | 5   | 查询时使用几个宇宙（影响召回/速度）。 |
| 随机种子                   | 0   | 固定超平面可保证可复现性。       |

**调参建议：**

* 增大 `N_PLANES`：提高区分度但降低命中率
* 增大 `N_UNIVERSES` / `num_universes_to_use`：提升召回但增加查询耗时
* 在高维场景通常优先调整宇宙数量，提高稳定性

---

## 6. 复杂度分析

### 构建阶段

```
O(M · D · N_PLANES · N_UNIVERSES)
```

主要来自向量投影与哈希构造。

### 查询阶段

* LSH 命中：`O(N_PLANES · U)`
* 候选重排：`O(|C| · D)`

其中 `U = num_universes_to_use`，`|C|` 为候选数量。

## 7. 运行结果示例

```
length of dictionary 10000
shape of document_vecs (10000, 300)
The hash value for a random vector with planes[0] is 512
working on hash universe #: 0
working on hash universe #: 1
working on hash universe #: 2
working on hash universe #: 3
working on hash universe #: 4
working on hash universe #: 5
working on hash universe #: 6
working on hash universe #: 7
working on hash universe #: 8
working on hash universe #: 9
working on hash universe #: 10
working on hash universe #: 11
working on hash universe #: 12
working on hash universe #: 13
working on hash universe #: 14
working on hash universe #: 15
working on hash universe #: 16
working on hash universe #: 17
working on hash universe #: 18
working on hash universe #: 19
working on hash universe #: 20
working on hash universe #: 21
working on hash universe #: 22
working on hash universe #: 23
working on hash universe #: 24
Nearest neighbors for document 0
Document contents: #FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)

Nearest neighbor at document id 51
document contents: #FollowFriday @France_Espana @reglisse_menthe @CCI_inter for being top engaged members in my community this week :)
Nearest neighbor at document id 105
document contents: #FollowFriday @straz_das @DCarsonCPA @GH813600 for being top engaged members in my community this week :)
Nearest neighbor at document id 154
document contents: #FollowFriday @IzywayLesExpats @na4innov @InXpressCoAzur for being top engaged members in my community this week :)
```