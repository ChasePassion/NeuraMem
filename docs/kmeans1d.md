# kmeans1d 使用指南

> 一维 k-means 全局最优聚类库，用于你的"记忆选择策略"

---

## 1. 库简介

`kmeans1d` 是一个专门做 **一维 k-means 聚类** 的 Python 库，底层用 C++ 实现，一维场景下能在 **多项式时间内给出全局最优解**。

关键特点：

- **只处理 1D 数据**：输入是一维数组，比如相似度分数、打分、时间等标量。
- **返回全局最优划分**：不是像标准 Lloyd 算法那样局部最优，而是 DP 算法保证全局最优。
- **时间复杂度**：$O(k n + n \log n)$，适合你这种 n=100 级别的小规模决策场景。
- **实现方式**：核心是 C++，通过 Python binding 暴露 `cluster()` 接口。

在你的记忆系统里，可以把"TOP100 相似度分数"看成一维数组，用 `kmeans1d` 自动找到 **高相关簇 vs 低相关簇**，再据此截断。

---

## 2. 安装与环境要求

### 2.1 环境要求

- Python 3.x（官方说明是 Python 3 系列）
- 支持的操作系统：Windows / macOS / Linux（三大平台都可以）。

### 2.2 安装

```bash
pip install kmeans1d
# 或者显式用 Python3
pip3 install kmeans1d
```

该包发布在 PyPI 上，直接通过 pip 安装即可。

---

## 3. 核心 API 概览

kmeans1d 的 API 极其克制，主要就一个函数：`cluster()`。

### 3.1 kmeans1d.cluster(array, k)

签名（从源码推断）：

```python
import kmeans1d
from typing import Sequence, List, Tuple

clusters, centroids = kmeans1d.cluster(array: Sequence[float], k: int)
```

底层实际返回的是一个 `namedtuple('Clustered', 'clusters centroids')`，Python 支持拆包，所以你可以直接写：

```python
clusters, centroids = kmeans1d.cluster(x, k)
```

#### 参数说明

**array: Sequence[float]**

一维数据序列，支持：

- Python list，如 `[0.95, 0.91, 0.88, ...]`
- NumPy array（最好 `array.tolist()` 一下）
- 任何可迭代的 float 序列

**k: int**

目标簇数

- 必须 > 0，否则会触发断言：`assert k > 0`
- 如果 k > len(array)，库内部会自动做：
  ```python
  k = min(k, n)
  ```
  即最多不会超过样本数。

#### 返回值说明

`cluster()` 返回一个二元结构 `(clusters, centroids)`：

**clusters: List[int]**

- 长度 = n，每个元素是该点所在的簇编号（0 ~ k-1）
- 注意：簇编号与数据顺序一一对应：`clusters[i]` 是 `array[i]` 的聚类编号。

**centroids: List[float]**

- 长度 = k
- 每个元素是对应簇的中心（均值）

---

## 4. 快速上手示例

官方示例：

```python
import kmeans1d

x = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
k = 4

clusters, centroids = kmeans1d.cluster(x, k)

print("clusters: ", clusters)
print("centroids:", centroids)
```

输出（可能类似）：

```text
clusters:  [1, 1, 1, 0, 3, 3, 3, 2, 2, 2]
centroids: [-50.0, 4.1, 94.0, 200.5]
```

直觉：

- -50 被单独分成一组（簇 0）；
- 4.x 附近的点是一组（簇 1）；
- 80、100、102 聚成一组（簇 2），中心约 94；
- 200.x 三个点聚成一组（簇 3）。

---

## 5. 与你的记忆系统结合使用的示例

你的场景：

- 有一批检索到的记忆（比如 top-100），每条有一个相似度分数（越大越相关）。
- 想用一个自适应的方法，在这些分数里把"真正相关的一簇"自动切出来，而不是拍脑袋设阈值。

这里可以把这些分数作为 1D 数据，使用 kmeans1d：

### 5.1 典型用法：k=2，高相关 vs 低相关

```python
import kmeans1d

def select_relevant_memories_by_kmeans(scores, memories, k=2):
    """
    scores:   List[float]，与 memories 一一对应的相似度
    memories: List[Memory]，你自己的记忆对象或 dict
    
    返回：被认为"高相关"的那一簇的记忆列表
    """
    # 1. 聚类
    clusters, centroids = kmeans1d.cluster(scores, k)

    # 2. 找到"均值最高"的簇，视为高相关簇
    #   对于相似度来说，均值越高簇越"相关"
    best_cluster = max(range(len(centroids)), key=lambda c: centroids[c])

    # 3. 选出属于该簇的所有记忆
    selected = [
        mem for mem, c in zip(memories, clusters)
        if c == best_cluster
    ]
    return selected, clusters, centroids
```

使用方式：

```python
scores   = [0.99, 0.95, 0.93, 0.90, 0.65, 0.64, 0.63, 0.40, 0.39, 0.10]
memories = [...]  # 与 scores 对应的记忆对象

selected, clusters, centroids = select_relevant_memories_by_kmeans(scores, memories, k=2)

print("centroids:", centroids)
print("high relevant count:", len(selected))
```

逻辑：

- 如果分数自然分为"高分一堆 + 低分一堆"，k=2 通常就能把"高分簇"单独聚出来；
- 比简单的"固定阈值 0.7"更自适应，因为分界点是由数据分布决定的。

### 5.2 更细：k=3 或以上，支持"中间层"

如果你担心只有"高 / 低"太粗，可以设 k=3：

- 一般会变成：高相关簇 / 中度相关簇 / 噪音簇；
- 然后你可以：
  - 一定保留"高相关簇"；
  - 根据数量或阈值，选择是否附带"中度相关簇"。

示意代码：

```python
def select_memories_with_middle_cluster(scores, memories, k=3):
    clusters, centroids = kmeans1d.cluster(scores, k)

    # 按中心从高到低排序簇
    order = sorted(range(k), key=lambda c: centroids[c], reverse=True)

    high_cluster = order[0]
    mid_cluster  = order[1]

    high_mems = [m for m, c in zip(memories, clusters) if c == high_cluster]
    mid_mems  = [m for m, c in zip(memories, clusters) if c == mid_cluster]

    return {
        "high": high_mems,
        "mid":  mid_mems,
        "all_clusters": clusters,
        "centroids": centroids,
    }
```