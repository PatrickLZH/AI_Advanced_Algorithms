# 🔄 Online模块

在线检索模块，负责搜索接口、搜索引擎、CTR收集和模型训练。

## 📁 文件说明

### `search_interface.py`
- **功能**: 定义搜索和CTR接口抽象类
- **职责**: 接口定义和规范

### `search_engine.py`
- **功能**: 搜索引擎实现
- **职责**: 
  - 实现SearchInterface
  - 倒排索引查询
  - TF-IDF召回算法
  - CTR模型排序算法
  - 文档获取

### `ctr_collector.py`
- **功能**: CTR数据收集器
- **职责**:
  - 实现CTRInterface
  - 曝光记录
  - 点击记录
  - 数据自动保存

### `ctr_model.py`
- **功能**: CTR模型训练和预测
- **职责**:
  - 特征工程
  - 逻辑回归模型训练
  - CTR分数预测
  - 模型持久化

### `ctr_lr_model.py`
- **功能**: CTR模型独立训练脚本
- **职责**: 独立的模型训练和评估

## 🔗 依赖关系

```
search_interface.py (接口定义)
    ↓
search_engine.py (搜索引擎)
    ↓
ctr_collector.py (数据收集)
    ↓
ctr_model.py (模型训练)
```

## 🚀 使用方式

```python
# 导入模块
from online.search_engine import SearchEngine
from online.ctr_collector import CTRCollector
from online.ctr_model import CTRModel

# 初始化组件
search_engine = SearchEngine()
ctr_collector = CTRCollector()
ctr_model = CTRModel()
``` 