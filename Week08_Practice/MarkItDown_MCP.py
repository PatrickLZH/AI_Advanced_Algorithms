import os
import requests
from markitdown import MarkItDown
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

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

# 定义PDF文件资源
pdf_resource = Resource(
    uri="file:///*.pdf",
    type=ResourceType.FILE,
    name="pdf_file",
    description="需要转换为Markdown格式的PDF文件",
    metadata={
        "file_type": "PDF",
        "purpose": "转换为Markdown格式",
        "target_format": "Markdown"
    }
)

# 添加资源到上下文
mcp_context.add_resource(pdf_resource)

# 4. 定义Tools（工具）

# 定义PDF转换工具 (基于MarkItDown MCP)
pdf_conversion_tool = Tool(
    name="pdf_to_markdown_converter",
    description="使用MarkItDown MCP将PDF文件转换为Markdown格式文档的工具",
    parameters={
        "type": "object",
        "properties": {
            "input_file": {
                "type": "string",
                "description": "输入的PDF文件路径"
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
mcp_context.add_tool(pdf_conversion_tool)

# 5. 定义Prompts（提示）
# 定义PDF转换提示 (基于MarkItDown MCP)
pdf_conversion_prompt = Prompt(
    name="pdf_conversion_prompt",
    content="""
    请使用MarkItDown MCP将指定的PDF文件转换为结构化的Markdown文档。
    输入文件: {input_file}
    输出文件: {output_file}
    
    转换要求：
    1. 保留原文档的结构和格式
    2. 正确处理标题、段落、列表等元素
    3. 尽可能保留图像描述（如果有的话）
    4. 保持文档的可读性和结构化
    
    请确保输出格式清晰、易读。
    """,
    role="user",
    resources=["pdf_file"]
)

# 添加提示到上下文
mcp_context.add_prompt(pdf_conversion_prompt)

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
        
        if request_type == "convert_pdf_to_markdown":
            return self._convert_pdf_to_markdown(parameters) 
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    def _convert_pdf_to_markdown(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """将PDF文件转换为Markdown格式"""
        input_file = parameters.get("input_file", "three_kingdoms.pdf")
        output_file = parameters.get("output_file", "three_kingdoms.md")
        
        # 使用资源和工具
        resources_used = [r for r in self.context.resources.values() 
                         if r.name in ["pdf_file"]]
        tools_used = [t for t in self.context.tools.values() 
                     if t.name in ["pdf_to_markdown_converter"]]
        
        try:
            # 检查输入文件是否存在
            if not os.path.exists(input_file):
                return {
                    "result": "PDF到Markdown转换失败",
                    "resources_used": [r.name for r in resources_used],
                    "tools_used": [t.name for t in tools_used],
                    "error": f"未找到输入文件: {input_file}"
                }
            
            # 调用MarkItDown MCP服务进行PDF转换
            # 注意：这需要实际运行MarkItDown MCP服务
            # 参考 https://github.com/microsoft/markitdown/tree/main/packages/markitdown-mcp
            markdown_content = self._call_markitdown_mcp_service(input_file)
            
            if markdown_content: 
                # 保存Markdown文件
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                return {
                    "result": "PDF到Markdown转换完成",
                    "resources_used": [r.name for r in resources_used],
                    "tools_used": [t.name for t in tools_used],
                    "output_file": output_file
                }
            else:
                return {
                    "result": "PDF到Markdown转换失败",
                    "resources_used": [r.name for r in resources_used],
                    "tools_used": [t.name for t in tools_used],
                    "error": "MarkItDown MCP服务返回空结果"
                }
        except Exception as e:
            return {
                "result": "PDF到Markdown转换失败",
                "resources_used": [r.name for r in resources_used] if 'resources_used' in locals() else [],
                "tools_used": [t.name for t in tools_used] if 'tools_used' in locals() else [],
                "error": str(e)
            }
    
    def _call_markitdown_mcp_service(self, input_file: str) -> str:
        """
        调用MarkItDown MCP服务进行PDF转换
        根据MarkItDown MCP的规范实现
        """
        try:
            # # 方法1: 尝试直接使用markitdown包（如果已安装）
            # try:
            #     print("使用本地markitdown包进行转换...")
            #     markitdown = MarkItDown()
            #     result = markitdown.convert(input_file)
                
            #     # 检查返回结果的属性
            #     if hasattr(result, 'text'):
            #         return result.text
            #     elif hasattr(result, 'markdown'):
            #         return result.markdown
            #     else:
            #         # 如果没有预期的属性，尝试转换为字符串
            #         return str(result)
            # except ImportError:
            #     print("未找到markitdown包")
            # except Exception as e:
            #     print(f"使用markitdown包时出错: {e}")

            # 方法2: 尝试使用HTTP进行转换
            try:
                print("尝试使用HTTP服务进行转换...")
                # 获取文件名
                file_name = os.path.abspath(input_file)
                print(f"文件名: {file_name}")
                # 使用MCP Inspector中显示的URL
                url = "http://localhost:3001/mcp"
                # 构造符合MCP规范的完整请求（基于MCP Inspector的示例）
                payload = {
                    "jsonrpc": "2.0",  # 添加必需的jsonrpc字段
                    "id": "1",  # 添加必需的id字段
                    "method": "tools/call",
                    "params": {
                        "name": "convert_to_markdown",
                        "arguments": {
                            "uri": f"file:///{file_name}"
                        },
                        "_meta": {
                            "progressToken": 0
                        }
                    }
                }               
                # 设置正确的请求头
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
                # 发送POST请求
                print("发送HTTP请求...")
                response = requests.post(url=url, json=payload, headers=headers)
                # 检查响应状态
                if response.status_code == 200:
                    json_response = response.json()                   
                    # 处理MCP响应格式
                    if 'result' in json_response:
                        result = json_response['result']
                        # 根据提供的响应格式提取文本内容
                        if 'content' in result and len(result['content']) > 0:
                            content_items = result['content']
                            # 合并所有文本内容
                            markdown_text = ""
                            for item in content_items:
                                if item.get('type') == 'text' and 'text' in item:
                                    markdown_text += item['text']  # 显示前100个字符
                            print(f"成功提取到{len(content_items)}个文本片段")
                            return markdown_text
                        else:
                            # 如果没有content字段，尝试其他可能的字段
                            text_content = result.get('text', str(result))
                            return text_content
                    else:
                        # 如果响应不是标准JSON-RPC格式，尝试直接解析
                        if 'content' in json_response and len(json_response['content']) > 0:
                            content_items = json_response['content']
                            # 合并所有文本内容
                            markdown_text = ""
                            for item in content_items:
                                if item.get('type') == 'text' and 'text' in item:
                                    markdown_text += item['text']
                            print(f"成功提取到{len(content_items)}个文本片段")
                            return markdown_text
                        else:
                            text_content = json_response.get('text', str(json_response))
                            return text_content           
                else:
                    # 打印完整的错误信息
                    error_message = f"HTTP服务错误: {response.status_code} - {response.text}"
                    print(error_message)
                    return error_message       
            except requests.exceptions.ConnectionError as e:
                print(f"连接错误: {e}")
                print("请检查MCP服务是否正在运行，并且端口3001是否可访问")
            except requests.exceptions.Timeout as e:
                print(f"超时错误: {e}")
            except requests.exceptions.RequestException as e:
                print(f"请求异常: {e}")
            except Exception as e:
                print(f"未预期的异常: {e}")
        except Exception as e:
            print(f"调用MarkItDown MCP服务时出错: {str(e)}")
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        return datetime.now().isoformat()

def convert_pdf_to_markdown_mcp():
    # 执行PDF到Markdown的转换流程
    print("2 PDF到Markdown转换流程")

    # 初始化处理器
    processor = ContextProcessor(mcp_context)
    
    # 1) 显示PDF转换相关组件
    print("1) MarkItDown MCP组件:")
    pdf_resources = [r for r in mcp_context.resources.values() if r.name == "pdf_file"]
    pdf_tools = [t for t in mcp_context.tools.values() if t.name == "pdf_to_markdown_converter"]
    pdf_prompts = [p for p in mcp_context.prompts.values() if p.name == "pdf_conversion_prompt"]
    
    if pdf_resources:
        res = pdf_resources[0]
        print(f"   资源: {res.name} ({res.type.value}) - {res.description}")
    
    if pdf_tools:
        tool = pdf_tools[0]
        print(f"   工具: {tool.name} - {tool.description}")
    
    if pdf_prompts:
        prompt = pdf_prompts[0]
        print(f"   提示: {prompt.name} (角色: {prompt.role})")
    
    # 2) 执行转换
    print("2) 执行PDF到Markdown转换:")
    conversion_result = processor.process_request("convert_pdf_to_markdown", {
        "input_file": "three_kingdoms.pdf",
        "output_file": "three_kingdoms.md"
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

    # 2. 执行PDF转换
    convert_pdf_to_markdown_mcp()

    # 3 显示上下文历史
    print("3 上下文历史记录:")
    for i, history in enumerate(mcp_context.context_history, 1):
        print(f"{i}) {history['request_type']} - {history['timestamp']}")

# 执行完整流程
if __name__ == "__main__":
    execute_mcp_workflow()