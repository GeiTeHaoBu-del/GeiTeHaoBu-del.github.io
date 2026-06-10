---
title: 当大模型服务挂了怎么办？LLM多模型故障转移实战
date: 2026-06-10
tags: [LLM, 故障转移, Python, 微服务韧性]
categories: [工程实践]
description: 从一次线上大模型宕机出发，设计并实现连续失败计数→自动切换备模型→新周期自动恢复主模型的故障转移机制，附带三级降级链与实验验证
---

## 1. 问题引入

我们的热搜监控系统每30秒爬取一次微博热榜，调用大模型对每条热搜做情感分析、类型分类和话题提取。某天凌晨，主模型API突然返回429限流，系统连续30分钟没有产出任何分析结果——前端图表空白、预警静默、数据库趋势断档。

这不是小概率事件。LLM服务不稳定几乎是常态：429限流、5xx宕机、API Key过期、响应超时，任何一个都能让单模型依赖的系统瘫痪。**单点依赖 = 单点故障**，而LLM的不可用不是"会不会"的问题，是"什么时候"的问题。

我们需要回答一个核心问题：**当LLM服务不可用时，如何保证系统持续产出分析结果？**

## 2. 原理讲解

### 2.1 故障转移的三层设计

我们的方案围绕三个核心机制展开：

**第一层：连续失败计数**。不是一次失败就切换——偶发的网络抖动不应触发模型切换。只有连续失败达到阈值（默认2次），才判定为模型级别的故障，触发切换。这避免了"狼来了"式的频繁切换。

**第二层：模型切换**。当连续失败达到阈值，系统沿 `primary → backup` 优先级链切换到下一个可用模型，同时重置失败计数器。每次切换都会记录时间戳、原模型、目标模型和原因，形成审计日志。

**第三层：自动恢复**。切换到备模型后，系统不会永远停留在备模型上。每当一个新的请求周期开始（即下一轮爬取），系统会尝试回到主模型。因为主模型宕机往往是临时的——限流窗口过了、服务恢复了——自动恢复确保系统不会因为一次临时故障而永久降级。

### 2.2 为什么不用完整的熔断器？

Circuit Breaker模式有三种状态：Closed → Open → Half-Open。它适合高频请求场景（每秒成百上千次调用），需要精确控制"探针"频率。但我们的场景不同：**每30秒才调用一次LLM**，请求频率极低。在这种低频场景下，完整熔断器的半开状态、探针窗口反而增加了不必要的复杂度。轻量的连续失败计数已经足够——每次爬取本身就是天然的检测点，无需额外定时器。

### 2.3 三级降级链：任何情况下都有输出

故障转移解决的是"换一个模型试试"，但若所有模型都不可用呢？我们设计了三级降级链：

```
批量API调用 → 单条API调用 → 返回默认值
```

- **批量API**：10条标题一次请求，效率最高
- **单条API**：批量失败时退化为逐条调用，可能部分成功
- **默认值**：所有API都失败时，返回 `sentiment_score=0, type_name='其他', topic_name='无'`，系统**永远不崩溃，永远有输出**

类比值班体系：**主治医师（主模型批量）→ 值班医生（备模型单条）→ 急救包（默认值）**，确保任何时刻都有救治能力。

## 3. 代码实现

### 3.1 核心状态机

故障转移的核心是三个状态变量的协同：`current_model`（当前使用的模型标识）、`consecutive_failures`（连续失败次数）、`max_retries`（触发切换的阈值）。

```python
# llm_client.py:38-76 初始化
def __init__(self, api_config):
    failover_config = api_config.get('failover', {})
    self.max_retries = failover_config.get('max_retries', 2)
    self.auto_recover = failover_config.get('auto_recover', True)

    self.models = {}
    model_order = ['primary', 'backup']
    for model_name in model_order:
        if model_name in api_config:
            self.models[model_name] = self._init_model_config(
                api_config[model_name], model_name
            )

    self.current_model = 'primary'
    self.consecutive_failures = 0
    self.switch_history = []
```

配置结构天然支持多模型——`primary` 和 `backup` 各自持有独立的 `api_url`、`api_key`、`model` 名称，切换时只需更换 `current_model` 指针。

### 3.2 chat()主循环：故障转移的指挥中心

```python
# llm_client.py:168-223 精简版
def chat(self, messages, temperature=0.3, max_tokens=2000):
    # ① 新周期尝试恢复主模型
    self._try_recover_primary()

    attempted_models = []
    while True:
        model_config = self.models.get(self.current_model)
        attempted_models.append(self.current_model)

        # ② 发起请求
        result = self._do_chat_request(model_config, messages, temperature, max_tokens)

        if result is not None:
            # ③ 成功：重置失败计数
            self.consecutive_failures = 0
            return result

        # ④ 失败：累加计数
        self.consecutive_failures += 1

        if self.consecutive_failures >= self.max_retries:
            # ⑤ 达到阈值：切换模型
            self._switch_model(f"连续失败{self.max_retries}次")
            # 所有模型都试过了 → 返回None
            if self.current_model in attempted_models:
                return None
        else:
            # ⑥ 未达阈值：线性退避等待
            time.sleep(1 * self.consecutive_failures)
```

控制流的精髓：步骤①是自动恢复的入口，步骤③⑤⑥构成"失败→重试→切换"的决策链，`attempted_models` 防止无限循环——如果所有模型都已尝试，立即返回None。

### 3.3 模型切换与审计

```python
# llm_client.py:126-158 精简版
def _switch_model(self, reason):
    next_model = self._get_next_model()
    if next_model:
        old_model = self.current_model
        self.current_model = next_model
        self.consecutive_failures = 0

        self.switch_history.append({
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'from': old_model,
            'to': next_model,
            'reason': reason
        })
```

`switch_history` 是生产环境排障的关键——当运维发现分析结果异常时，可以回溯"什么时候、从哪个模型、切到哪个模型、因为什么"，而不是对着黑盒猜测。

### 3.4 自动恢复：新周期的回归

```python
# llm_client.py:160-166
def _try_recover_primary(self):
    if self.auto_recover and self.current_model != 'primary' and 'primary' in self.models:
        self.current_model = 'primary'
        self.consecutive_failures = 0
```

仅7行代码，却是一个精妙的设计决策：**在每次新请求周期的开头尝试恢复**。为什么不是定时器？因为我们的爬取周期就是天然的检测点——下一轮调用时自然就知道主模型是否恢复了。没有额外线程、没有定时器、没有复杂状态，代码极简但效果确定。

### 3.5 批量→单条的退化

```python
# llm_analyzer.py:336-376 精简版
def _call_batch_api(self, titles):
    titles_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])
    messages = [
        {"role": "system", "content": self.batch_system_prompt},
        {"role": "user", "content": f"请批量分析以下{len(titles)}个热搜标题：\n\n{titles_text}"}
    ]

    content = self.client.chat(messages, temperature=0.2)
    if not content:
        # 批量失败 → 退化为逐条分析
        return [self.analyze(t) for t in titles]

    result = self.client.parse_json_response(content)
    if not result:
        # 解析失败 → 同样退化
        return [self.analyze(t) for t in titles]

    if isinstance(result, list):
        return self._normalize_results(result, titles)
    elif isinstance(result, dict):
        return [result] + [self._default_result(t) for t in titles[1:]]
    else:
        return [self._default_result(t) for t in titles]
```

退化逻辑的关键：**批量失败不是直接返回默认值，而是降级为单条调用**。单条调用会再次经过 `chat()` 的故障转移逻辑，这意味着每条标题都有独立的机会通过备模型获得分析结果。只有当单条调用也失败时，才会走到最终的 `_default_result()` 兜底。

### 3.6 最终兜底

```python
# llm_analyzer.py:452-467
def _default_result(self, title):
    return {
        'sentiment_score': 0.0,
        'type_name': '其他',
        'topic_name': '无',
        'keywords': []
    }
```

默认值的设计不是"空"或"None"，而是**语义合理的占位值**：情感分数为0（中性）、类型为"其他"、话题为"无"。下游系统无需特殊处理null值，前端图表正常渲染，只是精度降低而非完全中断。

## 4. 实验结论

### 4.1 基线性能

使用50条测试用例（覆盖娱乐/社会/科技/体育/时政5类，每类10条），以10条为一组批量调用，禁用缓存：

| 指标 | 结果 | 基准线 |
|------|------|--------|
| 调用成功率 | **96%** | ≥90% |
| 批量响应时间（10条） | **8.2s** | ≤30s |
| 情感分析准确率 | **88%** | ≥70% |
| 类型分类准确率 | **82%** | ≥70% |
| 话题提取成功率 | **94%** | ≥60% |

### 4.2 故障注入：主模型429

通过Mock `_do_chat_request` 使主模型固定返回None（模拟429限流），观察切换行为：

| 指标 | 结果 |
|------|------|
| 触发切换的请求数 | **2次**（与max_retries=2一致） |
| 切换后成功率 | **恢复至95%** |
| switch_history记录 | `{'from':'primary','to':'backup','reason':'连续失败2次'}` |

**关键发现**：主模型故障后，系统在第2次失败后自动切换到备模型，切换后成功率立即恢复。整个切换过程对调用方完全透明。

### 4.3 自动恢复验证

切换到backup后，恢复主模型可用性：

| 指标 | 结果 |
|------|------|
| 恢复到primary的周期数 | **1个周期**（下一次爬取即恢复） |
| 恢复后成功率 | **96%**（与基线一致） |

自动恢复机制工作正常：只要主模型恢复，下一个请求周期就会自动切回，无需人工干预。

### 4.4 全链路降级测试

模拟主模型和备模型均不可用：

| 降级层级 | 产出 |
|---------|------|
| 批量API | 失败 |
| 单条API | 失败 |
| 默认值 | **100%产出** |

系统在所有模型不可用时仍然持续产出默认值结果，**不崩溃、不中断**。数据精度降级，但服务连续性得到保证。

### 4.5 缓存对韧性的贡献

在稳态运行（大部分热搜跨轮次持续存在）下，Redis缓存命中率可达**80%以上**。这意味着即使LLM服务完全不可用，仍有80%的热搜条目能从缓存返回分析结果，只有新上榜的热搜才会使用默认值。缓存是故障转移之外的第二道防线。

## 5. 面试关联

- **面试题1："如何设计一个高可用的LLM调用方案？"** → 本文2-3节给出了完整答案：多模型配置 + 连续失败计数切换 + 新周期自动恢复 + 三级降级链，确保任何情况下系统有输出。

- **面试题2："熔断器模式和你的连续失败计数方案有什么区别？各适合什么场景？"** → 第2节分析了两者差异：熔断器适合高频请求（每秒百次级），需要半开状态精确控制探针频率；连续失败计数适合低频场景（每30秒一次），每次请求本身就是天然检测点，无需额外状态机。

- **面试题3："系统降级策略怎么设计？如何保证任何情况下都有输出？"** → 第3节的三级降级链是答案核心：批量→单条→默认值。设计原则是**语义合理的占位值优于null**，下游系统无需特殊处理，精度降低但服务不中断。