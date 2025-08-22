# 🚀 快速启动指南

## 一键启动（推荐）

```bash
python start_system.py
```

这个命令会自动：
1. 构建离线索引
2. 启动UI界面

## 手动启动

### 1. 构建索引
```bash
python offline/offline_index.py
```

### 2. 启动在线服务
```bash
python online/search_engine.py
```

### 3. 启动UI界面
```bash
python ui/portal.py
```

## 使用流程

1. **搜索**: 在界面中输入查询词（如"人工智能"）
2. **查看结果**: 点击结果行查看文档详情
3. **收集数据**: 系统自动记录曝光和点击数据
4. **导出数据**: 点击"导出CTR数据"按钮
5. **训练模型**: 在UI界面点击"训练CTR模型"按钮

## 示例查询

- 人工智能
- 机器学习
- 深度学习
- 神经网络
- 自然语言处理

## 系统架构

```
UI界面 (Gradio) ←→ 搜索引擎 (TF-IDF) ←→ 离线索引
```

## 故障排除

- 如果依赖缺失，运行 `pip install -r requirements.txt`
- 如果索引文件损坏，删除 `models/index_data.json` 重新构建 