# MCP 上下文报告

生成时间: 2025-08-16 10:19:12

## 资源 (Resources)

| 名称 | 类型 | 描述 |
|------|------|------|
| openweathermap_api | api | OpenWeatherMap天气API，用于获取实时天气信息 |
| json_context_file | file | 包含MCP上下文信息的JSON文件，用于转换为Markdown格式 |

## 工具 (Tools)

| 名称 | 描述 |
|------|------|
| openweathermap_query | 使用OpenWeatherMap查询实时天气信息的工具 |
| json_to_markdown_converter | 将JSON格式的MCP上下文数据转换为Markdown格式报告的工具 |

## 提示 (Prompts)

### openweathermap_weather_query

**角色**: user

**内容**:
```

    使用OpenWeatherMap API查询指定城市的实时天气信息：
    城市: {city}
    
    请使用OpenWeatherMap API获取并格式化天气信息。
    
```

### markdown_conversion_prompt

**角色**: user

**内容**:
```

    请将指定的JSON格式MCP上下文文件转换为结构化的Markdown报告。
    输入文件: {input_file}
    输出文件: {output_file}
    
    Markdown报告应包含以下部分：
    1. 标题和生成时间
    2. 资源列表（以表格形式展示）
    3. 工具列表（以表格形式展示）
    4. 提示列表（显示每个提示的详细内容）
    5. 历史记录（以表格形式展示）
    
    请确保输出格式清晰、易读。
    
```


## 历史记录 (History)

| 序号 | 请求类型 | 时间戳 |
|------|----------|--------|
| 1 | get_weather | 2025-08-16T10:19:09.001696 |
