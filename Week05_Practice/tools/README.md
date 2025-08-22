# 🛠️ Tools模块

工具模块，包含系统维护和演示数据生成工具。

## 📁 文件说明

### `reset_system.py`
- **功能**: 系统重置和维护工具
- **职责**:
  - 清理CTR数据和模型
  - 备份系统数据
  - 查看系统状态
  - 交互式操作

### `demo_data_generator.py`
- **功能**: 演示数据生成器
- **职责**:
  - 快速生成CTR演示数据
  - 模拟真实点击模式
  - 数据统计和验证
  - 教学演示支持

## 🚀 使用方式

### 系统重置
```bash
# 一键重置
python tools/reset_system.py reset

# 备份数据
python tools/reset_system.py backup

# 查看状态
python tools/reset_system.py status

# 交互模式
python tools/reset_system.py
```

### 演示数据生成
```bash
# 生成默认数据（10条）
python tools/demo_data_generator.py

# 生成指定数量数据
python tools/demo_data_generator.py 15
```

## 🔧 功能特性

### 重置工具
- **数据清理**: 删除CTR数据和模型文件
- **备份功能**: 时间戳备份重要数据
- **状态检查**: 查看系统文件状态
- **安全确认**: 防止误操作

### 数据生成器
- **智能生成**: 基于位置的点击模式
- **数据真实**: 模拟真实用户行为
- **快速演示**: 5分钟完成完整演示
- **教学友好**: 适合课堂演示

## 📊 数据格式

### CTR数据格式
```json
{
  "records": [
    {
      "timestamp": "2024-01-01 10:00:00",
      "query": "人工智能",
      "doc_id": "doc1",
      "position": 1,
      "score": 0.85,
      "clicked": 1,
      "summary": "文档摘要..."
    }
  ],
  "total_records": 10,
  "total_clicks": 3,
  "overall_ctr": 0.3
}
```

## 🎯 教学应用

### 快速演示流程
1. 生成演示数据（1分钟）
2. 启动系统（1分钟）
3. 查看数据（1分钟）
4. 训练模型（1分钟）
5. 体验效果（1分钟）

### 实验环境维护
- 课前：生成演示数据
- 课中：实时演示
- 课后：清理环境
- 备份：保存重要数据 