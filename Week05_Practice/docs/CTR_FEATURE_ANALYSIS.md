# CTR特征权重分析与修正

## 问题描述

用户发现CTR特征说明与实际权重不一致的问题：

### 原始描述（不准确）
- 位置特征: 搜索结果中的位置（最重要）
- 长度特征: 查询、文档、摘要长度
- 匹配度: 查询词在摘要中的匹配比例
- 历史CTR: 查询和文档的历史点击率
- 位置衰减: 位置越靠前CTR越高
- 相似度分数: 原始TF-IDF分数

### 实际权重排名
1. `position_decay`: 0.3568
2. `position`: 0.2233
3. `tfidf_score`: 0.2107
4. `match_score`: 0.1835
5. `doc_ctr`: 0.1562
6. `query_length`: 0.1498
7. `doc_length`: 0.1201
8. `summary_length`: 0.1201
9. `query_ctr`: 0.0467

## 问题分析

### 特征提取顺序
代码中的特征提取顺序为：
```python
features = np.hstack([
    position_features,           # 位置
    doc_lengths,                 # 文档长度
    query_lengths,               # 查询长度
    summary_lengths,             # 摘要长度
    match_scores,                # 查询匹配度
    query_ctr_features,          # 查询历史CTR
    doc_ctr_features,            # 文档历史CTR
    position_decay,              # 位置衰减
    df['score'].values.reshape(-1, 1)  # 原始相似度分数
])
```

### 特征名称映射
```python
feature_names = [
    'position', 'doc_length', 'query_length', 'summary_length',
    'match_score', 'query_ctr', 'doc_ctr', 'position_decay', 'tfidf_score'
]
```

## 修正措施

### 1. 更新UI特征说明
修正了 `examples/week3/ui/portal.py` 中的特征描述：

**修正后的特征说明：**
- **位置衰减**: 1/(位置+1)，位置越靠前权重越高（最重要）
- **位置特征**: 搜索结果中的绝对位置
- **相似度分数**: 原始TF-IDF分数
- **匹配度**: 查询词在摘要中的匹配比例
- **文档历史CTR**: 文档的历史点击率
- **查询长度**: 查询字符串长度
- **文档长度**: 文档摘要长度
- **摘要长度**: 摘要字符串长度
- **查询历史CTR**: 查询的历史点击率

### 2. 更新文档
修正了 `examples/week3/docs/CTR_USAGE.md` 中的特征工程描述：

**修正后的特征工程：**
- **位置特征**: position (绝对位置), position_decay (1/(位置+1))
- **相似度特征**: tfidf_score (原始TF-IDF分数)
- **匹配特征**: match_score (查询词匹配度)
- **历史CTR特征**: query_ctr, doc_ctr (历史点击率)
- **长度特征**: query_length, doc_length, summary_length

### 3. 更新代码注释
修正了 `examples/week3/online/ctr_model.py` 中的注释：
- 将"位置特征（最重要的CTR特征）"改为"位置特征（绝对位置）"
- 将"位置衰减特征（位置越靠前，CTR越高）"改为"位置衰减特征（位置越靠前，权重越高）"

## 特征重要性分析

### 最重要的特征
1. **position_decay (0.3568)**: 位置衰减特征，使用公式 1/(位置+1)
   - 位置1: 0.5
   - 位置2: 0.33
   - 位置3: 0.25
   - 位置4: 0.2
   - 位置5: 0.17

2. **position (0.2233)**: 绝对位置特征
   - 直接使用位置数字（1, 2, 3, ...）

3. **tfidf_score (0.2107)**: 原始相似度分数
   - 来自TF-IDF检索的原始分数

### 中等重要特征
4. **match_score (0.1835)**: 查询词匹配度
5. **doc_ctr (0.1562)**: 文档历史点击率
6. **query_length (0.1498)**: 查询长度

### 较低重要特征
7. **doc_length (0.1201)**: 文档长度
8. **summary_length (0.1201)**: 摘要长度
9. **query_ctr (0.0467)**: 查询历史点击率

## 结论

修正后的特征说明现在准确反映了实际的模型权重，其中：
- **位置衰减特征**是最重要的CTR预测因子
- **绝对位置特征**次之
- **相似度分数**和**匹配度**也具有重要影响
- **历史CTR特征**中，文档历史CTR比查询历史CTR更重要

这些修正确保了文档和代码的一致性，用户现在可以准确理解CTR模型的特征重要性。 