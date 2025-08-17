import os
import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')

# 1. 基础MCP设置
class ResourceType(Enum):
    FILE = "file"
    URL = "url"
    DATABASE = "database"
    API = "api"

@dataclass
class Resource:
    """资源定义"""
    uri: str
    type: ResourceType
    name: str
    description: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]

@dataclass
class Prompt:
    """提示定义"""
    name: str
    content: str
    role: str = "user"
    resources: Optional[List[str]] = None

# 2. 实现MCP核心组件
class MCPContext:
    """MCP上下文管理器"""
    
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.tools: Dict[str, Tool] = {}
        self.prompts: Dict[str, Prompt] = {}
        self.context_history: List[Dict] = []
    
    def add_resource(self, resource: Resource):
        """添加资源"""
        self.resources[resource.name] = resource
        print(f"Resource added: {resource.name}")
    
    def add_tool(self, tool: Tool):
        """添加工具"""
        self.tools[tool.name] = tool
        print(f"Tool added: {tool.name}")
    
    def add_prompt(self, prompt: Prompt):
        """添加提示"""
        self.prompts[prompt.name] = prompt
        print(f"Prompt added: {prompt.name}")
    
    def get_context(self) -> Dict[str, Any]:
        """获取当前上下文"""
        # 创建可JSON序列化的资源字典
        serializable_resources = {}
        for name, res in self.resources.items():
            res_dict = asdict(res)
            # 将ResourceType枚举转换为字符串
            res_dict['type'] = res.type.value
            serializable_resources[name] = res_dict
            
        return {
            "resources": serializable_resources,
            "tools": {name: asdict(tool) for name, tool in self.tools.items()},
            "prompts": {name: asdict(prompt) for name, prompt in self.prompts.items()},
            "history": self.context_history
        }

# 3. 定义Resources（资源）
# 初始化MCP上下文
mcp_context = MCPContext()

# 定义 Markdown 转换资源
markdown_resource = Resource(
    uri="file://./openweathermap_mcp_context.json",
    type=ResourceType.FILE,
    name="json_context_file",
    description="包含MCP上下文信息的JSON文件，用于转换为Markdown格式",
    metadata={
        "file_type": "JSON",
        "purpose": "MCP上下文数据",
        "target_format": "Markdown"
    }
)

# 添加资源到上下文
mcp_context.add_resource(markdown_resource)

# 4. 定义Tools（工具）

# 定义 Markdown 转换工具
markdown_conversion_tool = Tool(
    name="json_to_markdown_converter",
    description="将JSON格式的MCP上下文数据转换为Markdown格式报告的工具",
    parameters={
        "type": "object",
        "properties": {
            "input_file": {
                "type": "string",
                "description": "输入的JSON文件路径"
            },
            "output_file": {
                "type": "string",
                "description": "输出的Markdown文件路径"
            }
        }
    },
    required=["input_file", "output_file"]
)

# 添加工具到上下文
mcp_context.add_tool(markdown_conversion_tool)

# 5. 定义Prompts（提示）
# 定义 Markdown 转换提示
markdown_conversion_prompt = Prompt(
    name="markdown_conversion_prompt",
    content="""
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
    """,
    role="user",
    resources=["json_context_file"]
)

# 添加提示到上下文
mcp_context.add_prompt(markdown_conversion_prompt)

# 6. 实现上下文处理流程
class ContextProcessor:
    """上下文处理器"""
    
    def __init__(self, context: MCPContext):
        self.context = context
    
    def process_request(self, request_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求"""
        # 记录上下文历史
        history_entry = {
            "request_type": request_type,
            "parameters": parameters,
            "timestamp": self._get_timestamp()
        }
        self.context.context_history.append(history_entry)
        
        if request_type == "convert_to_markdown":
            return self._convert_to_markdown(parameters) 
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    def _convert_to_markdown(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """将JSON上下文转换为Markdown格式"""
        input_file = parameters.get("input_file", "openweathermap_mcp_context.json")
        output_file = parameters.get("output_file", "openweathermap_mcp_context.md")
        
        # 使用资源和工具
        resources_used = [r for r in self.context.resources.values() 
                         if r.name in ["json_context_file"]]
        tools_used = [t for t in self.context.tools.values() 
                     if t.name in ["json_to_markdown_converter"]]
        
        try:
            # 读取JSON文件
            with open(input_file, "r", encoding="utf-8") as f:
                context_data = json.load(f)
            
            # 转换为Markdown
            markdown_content = self._convert_context_to_markdown(context_data)
            
            # 保存Markdown文件
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            return {
                "result": "JSON到Markdown转换完成",
                "resources_used": [r.name for r in resources_used],
                "tools_used": [t.name for t in tools_used],
                "output_file": output_file
            }
        except FileNotFoundError:
            return {
                "result": "JSON到Markdown转换失败",
                "resources_used": [r.name for r in resources_used],
                "tools_used": [t.name for t in tools_used],
                "error": f"未找到输入文件: {input_file}"
            }
        except Exception as e:
            return {
                "result": "JSON到Markdown转换失败",
                "resources_used": [r.name for r in resources_used] if 'resources_used' in locals() else [],
                "tools_used": [t.name for t in tools_used] if 'tools_used' in locals() else [],
                "error": str(e)
            }
    
    def _convert_context_to_markdown(self, context_data: Dict[str, Any]) -> str:
        """将上下文数据转换为Markdown格式"""
        markdown_content = []
        
        # 添加标题
        markdown_content.append("# MCP 上下文报告")
        markdown_content.append("")
        markdown_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append("")
        
        # 添加资源部分
        markdown_content.append("## 资源 (Resources)")
        markdown_content.append("")
        if context_data.get("resources"):
            markdown_content.append("| 名称 | 类型 | 描述 |")
            markdown_content.append("|------|------|------|")
            for name, resource in context_data["resources"].items():
                res_type = resource.get("type", "")
                description = resource.get("description", "")
                markdown_content.append(f"| {name} | {res_type} | {description} |")
        else:
            markdown_content.append("无资源")
        markdown_content.append("")
        
        # 添加工具部分
        markdown_content.append("## 工具 (Tools)")
        markdown_content.append("")
        if context_data.get("tools"):
            markdown_content.append("| 名称 | 描述 |")
            markdown_content.append("|------|------|")
            for name, tool in context_data["tools"].items():
                description = tool.get("description", "")
                markdown_content.append(f"| {name} | {description} |")
        else:
            markdown_content.append("无工具")
        markdown_content.append("")
        
        # 添加提示部分
        markdown_content.append("## 提示 (Prompts)")
        markdown_content.append("")
        if context_data.get("prompts"):
            for name, prompt in context_data["prompts"].items():
                markdown_content.append(f"### {name}")
                markdown_content.append("")
                markdown_content.append(f"**角色**: {prompt.get('role', 'user')}")
                markdown_content.append("")
                markdown_content.append(f"**内容**:")
                markdown_content.append("```")
                markdown_content.append(prompt.get("content", ""))
                markdown_content.append("```")
                markdown_content.append("")
        else:
            markdown_content.append("无提示")
        markdown_content.append("")
        
        # 添加历史记录部分
        markdown_content.append("## 历史记录 (History)")
        markdown_content.append("")
        if context_data.get("history"):
            markdown_content.append("| 序号 | 请求类型 | 时间戳 |")
            markdown_content.append("|------|----------|--------|")
            for i, history in enumerate(context_data["history"], 1):
                request_type = history.get("request_type", "")
                timestamp = history.get("timestamp", "")
                markdown_content.append(f"| {i} | {request_type} | {timestamp} |")
        else:
            markdown_content.append("无历史记录")
        markdown_content.append("")
        
        return "\n".join(markdown_content)
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        return datetime.now().isoformat()

def convert_json_to_markdown_mcp():
    # 2 执行JSON到Markdown的转换流程
    print("2 Markdown转换流程")
    
    # 初始化处理器
    processor = ContextProcessor(mcp_context)
    
    # 1) 显示Markdown相关组件
    print("1) Markdown MCP组件:")
    markdown_resources = [r for r in mcp_context.resources.values() if r.name == "json_context_file"]
    markdown_tools = [t for t in mcp_context.tools.values() if t.name == "json_to_markdown_converter"]
    markdown_prompts = [p for p in mcp_context.prompts.values() if p.name == "markdown_conversion_prompt"]
    
    if markdown_resources:
        res = markdown_resources[0]
        print(f"   资源: {res.name} ({res.type.value}) - {res.description}")
    
    if markdown_tools:
        tool = markdown_tools[0]
        print(f"   工具: {tool.name} - {tool.description}")
    
    if markdown_prompts:
        prompt = markdown_prompts[0]
        print(f"   提示: {prompt.name} (角色: {prompt.role})")
    
    # 2) 执行转换
    print("2) 执行JSON到Markdown转换:")
    conversion_result = processor.process_request("convert_to_markdown", {
        "input_file": "openweathermap_mcp_context.json",
        "output_file": "openweathermap_mcp_context.md"
    })
    
    print(f"   结果: {conversion_result['result']}")
    if "output_file" in conversion_result:
        print(f"   输出文件: {conversion_result['output_file']}")
    if "error" in conversion_result:
        print(f"   错误: {conversion_result['error']}")
    
    print()

def execute_mcp_workflow():
    # 1. 显示当前上下文状态
    print("1 显示当前上下文状态:")
    context_state = mcp_context.get_context()
    print(f"   资源数量: {len(context_state['resources'])}")
    print(f"   工具数量: {len(context_state['tools'])}")
    print(f"   提示数量: {len(context_state['prompts'])}")
    print()

    # 2. 执行Markdown转换
    convert_json_to_markdown_mcp()

    # 3 显示上下文历史
    print("3 上下文历史记录:")
    for i, history in enumerate(mcp_context.context_history, 1):
        print(f"{i}) {history['request_type']} - {history['timestamp']}")

# 执行完整流程
if __name__ == "__main__":
    execute_mcp_workflow()

