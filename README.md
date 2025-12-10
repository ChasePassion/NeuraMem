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

## 心路历程

今年10月-12月基本处于萎靡不振的状态，提不起劲，于是想着看看书，期间看了一本书《认知天性》，我本想通过这本书来改善我的学习方法，但是读着读着突然想到，人工智能本质是人工机器模拟智能，那么现在存在的问题是AI没有记忆，要让AI跟人一样具备记忆我们参考人脑处理记忆的机制去搭建系统不就好了？于是开始和AI对话研究，并借助AI翻看了一些文章，最后有了这个项目，amazing，really amazing。不过项目目前还有很多缺陷，TodoList也很多，更致命的问题是提示词。相关的想法在我搭建系统的时候去了解到，其实已经非常常见，Langchain的记忆系统划分方式类似，且已经有相关的研究，目前前沿的研究走到了哪里还不清楚，是否会被机器学习/深度学习颠覆也不清楚，也许明天就能让AI把记忆写到参数里面呢......

---

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
uvicorn src.api.main:app --reload --port 8000
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
