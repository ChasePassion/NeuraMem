# NeuraMem

NeuraMem 是一个基于神经科学和记忆模型的人工智能记忆系统。它模仿人脑的记忆机制，将记忆划分为**情景记忆 (Episodic Memory)**、**语义记忆 (Semantic Memory)** 以及**叙事记忆 (Narrative Memory)**，旨在赋予 AI 更接近人类的记忆能力。

## 核心概念

项目参考认知心理学和神经科学，将记忆过程分为四个阶段，并将其映射到系统实现中：

| 记忆阶段 | 人脑机制 | NeuraMem 实现 |
| :--- | :--- | :--- |
| **编码 (Encoding)** | 将记忆写到海马体的过程，留下痕迹 | **写入情景记忆**：将从上下文中提取的信息存储为情景记忆 |
| **巩固 (Consolidation)** | 强化痕迹的过程 | **提取语义记忆**：从多个情景记忆中提炼出抽象的语义记忆 |
| **检索 (Retrieval)** | 通过痕迹提取知识 | **混合检索**：检索情景和语义记忆，并使用叙事记忆扩展上下文 |
| **再巩固 (Reconsolidation)** | 检索后改写记忆并强化痕迹 | **叙事链扩展**：将记忆添加到叙事记忆链中，形成连贯的主题叙事 |

### 记忆类型
*   **情景记忆 (Episodic Memory)**: 具体的事件记录（例如：用户昨天下午腹泻）。
*   **语义记忆 (Semantic Memory)**: 从事件中提炼出的抽象知识或结论（例如：用户容易腹泻）。
*   **叙事记忆 (Narrative Memory)**: 同一主题的事件叙事链（例如：用户腹泻 -> 找不到厕所 -> 拉裤子）。


## 评测结果 (Benchmark Results)

NeuraMem 在 [LoCoMo](https://github.com/snap-stanford/locomo) 长对话记忆基准上的评测结果（排除 adversarial 类题目；工作流定义见 [RUN_RECORD.md](neuramem_benchmark/RUN_RECORD.md) 第 6.1 节）：

**W3 完整记忆闭环 + MiniMax-M3**（2026-08-17，1537 题）：

| 类别 | 题目数 | 正确 | 准确率 |
| :--- | ---: | ---: | ---: |
| 1-multi-hop (多跳推理) | 280 | 201 | 71.79% |
| 2-temporal (时间推理) | 321 | 198 | 61.68% |
| 3-open-domain (开放域) | 96 | 58 | 60.42% |
| 4-single-hop (单跳问答) | 840 | 606 | 72.14% |
| **总体** | **1537** | **1063** | **69.16%** |

**W1 episodic-only 基线 + deepseek-chat**（2026-08-15，1540 题）：

| 类别 | 题目数 | 正确 | 准确率 |
| :--- | ---: | ---: | ---: |
| 1-multi-hop (多跳推理) | 282 | 160 | 56.74% |
| 2-temporal (时间推理) | 321 | 152 | 47.35% |
| 3-open-domain (开放域) | 96 | 47 | 48.96% |
| 4-single-hop (单跳问答) | 841 | 480 | 57.07% |
| **总体** | **1540** | **839** | **54.48%** |

**评测口径**：同一份 LoCoMo 数据（10 个会话）。W3 = ingest（manage + 每 7 session consolidate）→ 混合检索（episodic top-5 + 叙事组扩展 + semantic 全量）→ LLM 回答 → usage judge + 叙事分组（reconsolidation）→ LLM 判分（lenient 模板），全链路 MiniMax-M3，平均单题时延 19.11s；同时输出 KV cache（前缀缓存）命中率（记忆系统口径 19.06%）。W1→W3 总体提升 +14.68pp（工作流与模型同时变更，增益不可单独归因）。完整的环境配置、复现命令与 OpenViking 口径差异说明见 [neuramem_benchmark/RUN_RECORD.md](neuramem_benchmark/RUN_RECORD.md)。

一键复现：

```bash
python -m neuramem_benchmark.run_benchmark --all-samples --threads 10 --ingest-parallel 4 --output result/locomo_neuramem_all_results.csv
```

## 功能实现 (Feature Implementation)

项目通过以下核心类实现记忆系统的关键功能：

*   **Memory (System Facade)**: 系统的统一入口，封装了底层的复杂性，协调各个管理器组件工作。
*   **EpisodicMemoryManager**: 负责情景记忆的生命周期管理，分析对话上下文并执行记忆的添加、更新或删除。
*   **NarrativeMemoryManager**: 专注于叙事性记忆的高级管理，负责记忆的聚类（Clustering）和叙事组的维护。
*   **SemanticWriter**: 负责“学习”过程，定期从散碎的情景记忆中提炼出抽象的语义事实（Semantic Memory）。
*   **MemoryUsageJudge**: 在检索后评估记忆对当前上下文的有效性，优化检索质量。

## API 接口 (API Interface)

系统提供了一系列 RESTful API 来管理记忆和进行对话：

*   `POST /v1/chat`: 基于记忆增强的流式对话接口 (SSE)。
*   `POST /v1/memories/manage`: 根据对话内容智能添加、更新或删除记忆。
*   `POST /v1/memories/search`: 混合检索记忆，支持叙事组递归扩展。
*   `POST /v1/memories/consolidate`: 触发记忆整合任务。
*   `DELETE /v1/memories/reset`: 清空指定用户的所有记忆。

## 快速开始 (Quick Start)

### 前置要求 (Prerequisites)
*   Python 3.10+
*   [Milvus](https://milvus.io/) 向量数据库 (推荐使用 Docker 安装)

### 安装 (Installation)

1.  克隆仓库：
    ```bash
    git clone https://github.com/your-username/NeuraMem.git
    cd NeuraMem
    ```

2.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

### 配置 (Configuration)

1.  复制环境变量示例文件：
    ```bash
    cp .env.example .env
    ```

2.  编辑 `.env` 文件，配置您的 LLM API Key (如 OpenAI 或 DeepSeek) 和 Milvus 连接信息。

### 运行 (Run)

启动 Gradio 演示应用：
```bash
python demo/app.py
```

## API 使用 (API Usage)

### 1. 启动 API 服务
使用 `uvicorn` 启动 FastAPI 服务：
```bash
uvicorn neuramem_server.app:app --reload --port 8000
```
启动后，您可以访问 **Swagger UI** 查看完整文档：[http://localhost:8000/docs](http://localhost:8000/docs)

### 2. 调用示例 (Python)
使用 `requests` 库调用搜索接口：

```python
import requests

url = "http://localhost:8000/v1/memories/search"
payload = {
    "user_id": "test_user",
    "query": "我刚才说了什么？"
}

response = requests.post(url, json=payload)
print(response.json())
```

---

> **注意**: 项目具体结构与程序逻辑可在 [project_summary.md](project_summary.md) 文件中查看
