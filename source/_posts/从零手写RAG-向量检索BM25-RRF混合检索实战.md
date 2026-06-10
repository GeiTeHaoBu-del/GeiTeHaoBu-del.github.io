---
title: 从零手写RAG：向量检索+BM25+RRF混合检索实战
date: 2026-06-09
tags:
  - RAG
  - FAISS
  - BM25
  - RRF
  - 混合检索
categories:
  - RAG
summary: 用不到300行Python从零实现RAG系统，覆盖向量检索、BM25关键词检索、RRF混合融合、CrossEncoder重排序的完整pipeline，并给出每一步的原理和代码。
description: 用不到300行Python从零实现RAG系统，覆盖向量检索、BM25关键词检索、RRF混合融合、CrossEncoder重排序的完整pipeline。
---

## 问题引入：为什么需要混合检索？

大模型的回答质量，取决于它"看到"了什么。RAG（Retrieval-Augmented Generation）的核心思路很简单：先把相关文档找出来，再让大模型基于文档回答。

问题出在"找"这一步。我们有两种主流检索方式，各有盲区：

| 检索方式 | 擅长 | 盲区 |
|---------|------|------|
| 向量检索 | 语义相似——查"注意力机制"能找到含"self-attention"的文档 | 精确关键词匹配弱——查"2025年复试细则"可能漏掉 |
| BM25检索 | 关键词精确匹配——查"复试细则"精准命中 | 语义理解弱——查"深度学习入门"找不到含"神经网络基础"的文档 |

**单用任一路都有漏检风险，混合检索是互补方案。** 本文用不到300行Python，从零实现一个完整的RAG系统，覆盖：文档切分 → Embedding → BM25 + 向量双路检索 → RRF融合 → CrossEncoder重排序 → LLM生成。

> 完整代码见 [MiniRAG](https://github.com/GeiTeHaoBu-del/MiniRAG)

## 原理讲解

### 1. 向量检索：语义相似的"GPS定位"

Embedding模型把文本映射为256维向量。语义相近的文本，向量在高维空间中也相近——就像GPS坐标，两个点越近，物理距离越短。

```python
# 智谱 embedding-3：文本 → 256维向量
result = embedding_client.embeddings.create(
    model="embedding-3",
    input=text,
    dimensions=256
)
vector = result.data[0].embedding  # [0.12, -0.34, 0.56, ...]
```

向量检索的三步流程：查询向量化 → L2距离计算 → 取最近邻。

```python
# FAISS IndexFlatL2：暴力遍历，计算查询向量与所有文档向量的L2距离
index = faiss.IndexFlatL2(256)    # 256维
index.add(doc_vectors)            # 加入所有文档向量
D, I = index.search(query_vec, k=10)  # D=距离, I=索引
```

L2距离公式：$L2 = \sqrt{\sum(v1[i] - v2[i])^2}$，距离越小越相似。

`IndexFlatL2`是暴力遍历，适合小数据量（<10万条）。大数据量场景需要IVF或HNSW索引，这个后续文章再展开。

### 2. BM25：关键词匹配的数学升级版

BM25是TF-IDF的改进版，三个核心要素：

- **TF（词频）**：关键词在文档中出现次数越多，分越高——但有饱和机制（k1=1.5），出现10次和100次差距不大
- **IDF（逆文档频率）**：越稀有的词权重越高，"的"权重低，"Transformer"权重高
- **文档长度归一化**（b=0.75）：惩罚过长的文档，避免长文档因词多而占便宜

$$score(D,Q) = \sum IDF(q_i) \times \frac{f(q_i,D) \times (k_1+1)}{f(q_i,D) + k_1 \times (1-b+b \times |D|/avgdl)}$$

中文文本需要先分词，这里用jieba：

```python
import jieba
from rank_bm25 import BM25Okapi

tokenized_corpus = [jieba.lcut(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)           # k1=1.5, b=0.75
scores = bm25.get_scores(jieba.lcut(query))  # 计算所有文档的BM25分数
```

### 3. RRF融合：让两路检索"投票"

BM25分数和L2距离量纲不同、方向不同，不能直接相加。RRF（Reciprocal Rank Fusion）的思路：**不看原始分数，只看排名。**

$$score(d) = \sum \frac{1}{k + rank}$$

k=60是原论文推荐值，作用是平滑排名差异。k太小则排名1远优于排名2，k太大则排名差异被抹平。

举个具体例子：

| 文档 | BM25排名 | 向量检索排名 | RRF分数 |
|------|---------|------------|---------|
| 文档A | 1 | 2 | 1/61 + 1/62 = 0.0323 |
| 文档B | 3 | 1 | 1/63 + 1/61 = 0.0322 |
| 文档C | 2 | 未出现 | 1/62 + 0 = 0.0161 |

两路都推荐的文档A和B分数最高，只被一路推荐的C明显偏低——这就是"投票"的效果。

### 4. CrossEncoder重排序：最后一把精准筛

向量检索是Bi-Encoder：query和doc分别编码，速度快但精度有限。CrossEncoder把query和doc拼接后一起输入模型，精度高但速度慢。

策略：先用双路检索+RRF从大量文档中粗筛出5条，再用CrossEncoder精排取3条，送入LLM。

## 代码实现

### Pipeline总览

```
用户输入 → Query Rewrite → BM25检索(10条) + 向量检索(10条)
                              ↓
                        RRF融合(5条) → CrossEncoder重排序(3条) → LLM生成
```

### 1. 文档切分与Embedding

```python
def read_pdf_file(file_path, chunk_size=0, overlap=0):
    """PDF文本提取 + 滑动窗口切分"""
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    if chunk_size == 0:
        return text
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def embedding_invoke(text, config):
    """调用智谱embedding-3获取256维向量"""
    result = embedding_client.embeddings.create(
        model=config.embedding_name,
        input=text,
        dimensions=config.embedding_dim
    )
    return result.data[0].embedding
```

切分策略是固定长度滑动窗口（chunk_size=1000, overlap=200），优点是简单可控，缺点是可能在句子中间截断。改进方案后续会讨论。

### 2. BM25检索

```python
def BM25_search(user_query, database_content, config):
    tokenized_query = jieba.lcut(user_query)
    tokenized_corpus = [
        jieba.lcut(doc["content"])
        for doc in database_content.values()
    ]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(scores)[-config.bm25_top_k:][::-1]
    return top_k_indices, scores[top_k_indices]
```

### 3. 向量检索

```python
def embedding_search(user_query, database_index, config):
    query_embedding = embedding_invoke(user_query, config)
    query_vec = np.array([query_embedding], dtype=np.float32)
    index = faiss.IndexFlatL2(config.embedding_dim)
    index.add(database_index)
    D, I = index.search(query_vec, config.embedding_top_k)
    return D, I
```

### 4. RRF融合

```python
def RRF(bm25_result, embedding_result):
    k = 60
    scores = {}
    for rank, index in enumerate(bm25_result):
        scores[index] = scores.get(index, 0) + 1 / (k + rank + 1)
    for rank, index in enumerate(embedding_result[0]):
        scores[index] = scores.get(index, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 5. CrossEncoder重排序

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

def ReRanker_search(user_query, database_content, fusion_scores, config):
    top_docs = fusion_scores[:config.rrf_top_k]
    results = []
    for index, rrf_score in top_docs:
        doc_text = database_content[index]["content"]
        score = reranker.compute_score([user_query, doc_text])
        results.append((index, score))
    return sorted(results, key=lambda x: x[1], reverse=True)[:config.reranker_top_k]
```

`bge-reranker-v2-m3`是24层的XLMRoberta CrossEncoder，支持中英文，FP16推理。它对(query, doc)拼接输入，输出一个相关性分数，比Bi-Encoder更精准但更慢。

### 6. Query Rewriting与LLM生成

```python
def rewrite_query(messages, current_message, config):
    """用LLM+对话历史改写查询，解决代词指代问题"""
    rewritten = llm_invoke(
        rewrite_prompt.format(history=messages, question=current_message),
        config
    )
    return rewritten
```

Query Rewriting的价值：用户追问"它的参数怎么调？"时，LLM结合历史对话将其改写为"BM25的k1参数怎么调？"，这样检索才能命中相关文档。

最终用改写后的查询走完检索pipeline，取top-3文档拼接上下文，送入LLM生成回答。

## 实验结论与反思

### 检索漏斗效果

本系统的检索漏斗：10+10 → RRF 5 → Reranker 3 → LLM。逐级减少候选数，逐级增加计算成本——这是RAG系统的经典设计模式。

### 待改进点

| 问题 | 原因 | 改进方向 |
|------|------|----------|
| 切分可能在句中截断 | 固定长度切分，不感知句子边界 | 递归切分 / 语义切分 |
| BM25和FAISS索引每次重建 | 未持久化 | 预构建并保存索引 |
| 检索效果无量化评估 | 未做对比实验 | 设计三路检索对比实验 |
| 只支持PDF | 文档加载器单一 | 扩展DOCX/TXT/MD等格式 |
| API Key硬编码 | 安全隐患 | 环境变量 / .env文件 |

### 对比实验方案（后续可做）

用同一批查询，对比三种检索策略的召回率：

| 策略 | 方法 | 预期 |
|------|------|------|
| 纯向量 | FAISS top-10 | 语义相关但关键词弱 |
| 纯BM25 | BM25 top-10 | 关键词精准但语义弱 |
| 混合+RRF+Reranker | 完整pipeline | 综合最优 |

评估指标：Recall@5、MRR、人工判断相关性。建议用5-10个测试查询，每个查询标注3-5个相关文档，计算各策略的召回率。

## 面试关联

1. **RAG的工作流程？** → 检索+增强+生成三阶段，本文实现了完整的五步pipeline
2. **为什么需要混合检索？** → 向量检索和BM25互补，单用任一路有盲区
3. **RRF的原理？** → 只用排名不用原始分数，解决量纲不一致问题
4. **CrossEncoder vs Bi-Encoder？** → 前者精度高速度慢（适合精排），后者速度快精度低（适合粗筛）
5. **为什么需要Query Rewriting？** → 追问场景下代词指代问题，改写后才能检索到正确文档
