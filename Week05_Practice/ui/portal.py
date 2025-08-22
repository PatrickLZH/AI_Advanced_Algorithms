#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI模块 - 使用接口解耦
通过接口调用搜索引擎和CTR收集器，不依赖服务调用
"""

import gradio as gr
import json
import pandas as pd
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from online.search_engine import SearchEngine
from online.ctr_collector import CTRCollector
from online.ctr_model import CTRModel

class SearchUI:
    """搜索界面类 - 使用接口解耦"""
    
    def __init__(self):
        # 通过接口初始化组件
        self.search_engine = SearchEngine()
        self.ctr_collector = CTRCollector()
        self.ctr_model = CTRModel()
        self.current_query = ""
        self.setup_ui()
    
    def setup_ui(self):
        """设置Gradio用户界面"""
        with gr.Blocks(title="倒排索引检索系统") as self.interface:
            gr.Markdown("""
            # 🔍 倒排索引检索系统
            
            **教学演示系统：**
            - 支持TF-IDF召回、CTR排序、点击采集、特征可视化等功能
            - 推荐实验流程：搜索→点击→训练CTR→切换排序对比
            - 排序方式可在下方切换，体验不同排序效果
            - 训练后可查看CTR特征权重
            """)
            
            # 排序方式切换
            sort_mode = gr.Dropdown([
                ("TF-IDF排序", "tfidf"),
                ("CTR排序", "ctr")
            ], value="ctr", label="排序方式", info="选择排序方式：TF-IDF/CTR")
            
            with gr.Row():
                with gr.Column(scale=3):
                    # 搜索输入
                    query_input = gr.Textbox(
                        label="搜索查询",
                        placeholder="请输入要搜索的关键词...",
                        lines=1
                    )
                    
                    with gr.Row():
                        search_btn = gr.Button("🔍 搜索", variant="primary")
                        stats_btn = gr.Button("📊 显示统计信息")
                    
                    # 搜索结果
                    results_output = gr.Dataframe(
                        headers=["文档ID", "相似度分数", "文档长度", "摘要"],
                        label="搜索结果",
                        interactive=True,
                        wrap=True
                    )
                
                with gr.Column(scale=2):
                    # 文档内容显示
                    doc_content = gr.HTML(
                        label="文档内容",
                        value="<p>请输入搜索查询查看结果...</p>"
                    )
            
            # 示例查询
            with gr.Accordion("💡 示例查询", open=False):
                gr.Markdown("""
                可以尝试以下查询：
                - 人工智能
                - 机器学习  
                - 深度学习
                - 神经网络
                - 自然语言处理
                - 计算机视觉
                - 强化学习
                - 知识图谱
                """)
            
            # 查看历史记录
            with gr.Accordion("📚 查看历史记录", open=False):
                # 历史记录
                history_output = gr.HTML(
                    label="历史记录",
                    value=self.get_history_html()
                )
                
                # CTR数据状态
                ctr_status = gr.HTML(
                    label="CTR数据状态",
                    value="<p>系统会自动保存CTR数据到 data/ctr_data.json 文件</p>"
                )
            
            # CTR模型训练
            with gr.Accordion("🤖 CTR模型训练", open=False):
                with gr.Row():
                    train_btn = gr.Button("🚀 训练CTR模型", variant="primary")
                    model_status = gr.HTML(
                        label="模型状态",
                        value="<p>CTR模型未训练</p>"
                    )
                
                # 训练结果
                train_result = gr.HTML(
                    label="训练结果",
                    value="<p>点击训练按钮开始训练...</p>"
                )
                
                # 特征权重可视化
                feature_weights = gr.HTML(
                    label="特征权重",
                    value="<p>训练后可查看特征重要性...</p>"
                )
            
            # 绑定事件
            search_btn.click(
                fn=self.perform_search,
                inputs=[query_input, sort_mode],
                outputs=[results_output, doc_content, history_output]
            )
            
            stats_btn.click(
                fn=self.show_stats,
                outputs=[doc_content]
            )
            
            # 回车键搜索
            query_input.submit(
                fn=self.perform_search,
                inputs=[query_input, sort_mode],
                outputs=[results_output, doc_content, history_output]
            )
            
            # 表格行点击事件
            results_output.select(
                fn=self.on_row_select,
                outputs=[doc_content, history_output]
            )
            

            
            # 训练CTR模型
            train_btn.click(
                fn=self.train_ctr_model,
                outputs=[model_status, train_result, feature_weights]
            )
    
    def perform_search(self, query: str, sort_mode: str = "ctr"):
        """执行搜索 - 通过接口调用"""
        if not query.strip():
            return [], "请输入搜索查询", self.get_history_html()
        
        try:
            # 先retrieve召回文档ID
            doc_ids = self.search_engine.retrieve(query.strip(), top_k=20)
            # 再rank精排
            results = self.search_engine.rank(query.strip(), doc_ids, top_k=10)
            
            # 排序方式切换
            if sort_mode == "tfidf":
                # 只按TF-IDF分数排序
                results = sorted(results, key=lambda x: x[1], reverse=True)
            elif sort_mode == "ctr":
                # 有CTR分数时按CTR分数排序，否则按TF-IDF
                if any(x[2] is not None for x in results):
                    results = sorted(results, key=lambda x: (x[2] if x[2] is not None else 0), reverse=True)
                else:
                    results = sorted(results, key=lambda x: x[1], reverse=True)
            
            if not results:
                return [], "未找到相关文档", self.get_history_html()
            
            # 保存当前查询
            self.current_query = query.strip()
            
            # 记录CTR数据 - 所有展示的文档
            for position, result in enumerate(results):
                if len(result) == 4:  # 新格式：(doc_id, tfidf_score, ctr_score, summary)
                    doc_id, tfidf_score, ctr_score, summary = result
                else:  # 旧格式兼容：(doc_id, score, summary)
                    doc_id, tfidf_score, summary = result
                    ctr_score = None
                
                # 保存纯文本摘要用于CTR数据
                plain_summary = summary.replace('<span style="background-color: yellow; font-weight: bold;">', '').replace('</span>', '')
                self.ctr_collector.record_impression(
                    query=self.current_query,
                    doc_id=doc_id,
                    position=position + 1,
                    score=tfidf_score,  # 使用TF-IDF分数记录
                    summary=plain_summary
                )
            
            # 格式化结果
            formatted_results = []
            for result in results:
                if len(result) == 4:  # 新格式：(doc_id, tfidf_score, ctr_score, summary)
                    doc_id, tfidf_score, ctr_score, summary = result
                else:  # 旧格式兼容：(doc_id, score, summary)
                    doc_id, tfidf_score, summary = result
                    ctr_score = None
                
                doc_length = len(self.search_engine.get_document(doc_id))
                # 将HTML摘要转换为纯文本用于表格显示
                plain_summary = summary.replace('<span style="background-color: yellow; font-weight: bold;">', '').replace('</span>', '')
                
                # 显示两种分数
                if ctr_score is not None:
                    score_label = f"TF-IDF: {tfidf_score:.4f} | CTR: {ctr_score:.4f}"
                else:
                    score_label = f"TF-IDF: {tfidf_score:.4f}"
                
                formatted_results.append([
                    doc_id,
                    score_label,
                    doc_length,
                    plain_summary
                ])
            
            # 显示第一个文档的内容
            first_doc_id, first_tfidf_score, first_ctr_score, first_summary = results[0]
            first_content = self.search_engine.get_document(first_doc_id)
            
            html_content = f"""
            <div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <h3>文档ID: {first_doc_id}</h3>
                <div style="background-color: #f5f5f5; padding: 10px; border-left: 4px solid #007bff; margin: 10px 0;">
                    <strong>摘要:</strong><br>
                    {first_summary}
                </div>
                <div style="margin-top: 20px;">
                    <strong>完整文档内容:</strong><br>
                    {first_content}
                </div>
            </div>
            """
            
            return formatted_results, html_content, self.get_history_html()
            
        except Exception as e:
            error_msg = f"搜索失败: {e}"
            return [], error_msg, self.get_history_html()
    
    def on_row_select(self, evt: gr.SelectData):
        """处理表格行选择事件"""
        try:
            # 获取当前搜索结果
            current_results = self.search_engine.get_current_results()
            
            if not current_results or evt.index[0] >= len(current_results):
                return "文档不存在", self.get_history_html()
            
            # 获取选中的文档
            doc_id, tfidf_score, ctr_score, summary = current_results[evt.index[0]]
            content = self.search_engine.get_document(doc_id)
            
            # 记录点击行为
            self.ctr_collector.record_click(
                query=self.current_query,
                doc_id=doc_id,
                position=evt.index[0] + 1
            )
            
            # 生成文档内容
            html_content = f"""
            <div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <h3>文档ID: {doc_id}</h3>
                <div style="background-color: #f5f5f5; padding: 10px; border-left: 4px solid #007bff; margin: 10px 0;">
                    <strong>摘要:</strong><br>
                    {summary}
                </div>
                <div style="margin-top: 20px;">
                    <strong>完整文档内容:</strong><br>
                    {content}
                </div>
            </div>
            """
            
            return html_content, self.get_history_html()
            
        except Exception as e:
            error_msg = f"获取文档详情失败: {e}"
            return error_msg, self.get_history_html()
    
    def get_history_html(self):
        """获取历史记录"""
        records = self.ctr_collector.get_history()
        
        if not records:
            return "<p>暂无曝光历史...</p>"
        
        html_content = """
        <div style='font-family: Arial, sans-serif;'>
            <h3>📚 曝光历史（含点击label）</h3>
            <div style='max-height: 400px; overflow-y: auto;'>
        """
        
        for record in records:
            label = "<span style='color: #fff; background: #28a745; padding:2px 8px; border-radius:4px;'>点击</span>" if record['clicked'] else "<span style='color: #fff; background: #6c757d; padding:2px 8px; border-radius:4px;'>未点击</span>"
            html_content += f"""
            <div style='border: 1px solid #ddd; margin: 5px 0; padding: 10px; border-radius: 5px; background-color: #f9f9f9;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <strong>查询: {record['query']}</strong>
                    <span style='color: #666; font-size: 0.9em;'>{record['timestamp']}</span>
                </div>
                <div style='margin: 4px 0 4px 0;'>
                    <strong>文档ID:</strong> {record['doc_id']} | <strong>位置:</strong> #{record['position']} | <strong>分数:</strong> {record['score']:.4f}
                </div>
                <div style='margin: 4px 0 4px 0;'>
                    {label}
                </div>
                <div style='font-size: 0.9em; color: #555;'>
                    {record['summary'][:80]}...
                </div>
            </div>
            """
        
        html_content += """
            </div>
        </div>
        """
        return html_content
    
    def show_stats(self):
        """显示统计信息"""
        try:
            # 获取索引统计
            index_stats = self.search_engine.get_stats()
            
            # 获取CTR统计
            ctr_stats = self.ctr_collector.get_stats()
            
            html_content = f"""
            <div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <h2>📊 系统统计信息</h2>
                
                <h3>索引信息</h3>
                <ul>
                    <li><strong>总文档数</strong>: {index_stats['total_documents']}</li>
                    <li><strong>总词项数</strong>: {index_stats['total_terms']}</li>
                    <li><strong>平均文档长度</strong>: {index_stats['average_doc_length']:.2f}</li>
                </ul>
                
                <h3>CTR统计</h3>
                <ul>
                    <li><strong>总曝光次数</strong>: {ctr_stats['total_impressions']}</li>
                    <li><strong>总点击次数</strong>: {ctr_stats['total_clicks']}</li>
                    <li><strong>整体CTR</strong>: {ctr_stats['overall_ctr']:.4f} ({ctr_stats['overall_ctr']*100:.2f}%)</li>
                </ul>
            </div>
            """
            return html_content
            
        except Exception as e:
            return f"<p>获取统计信息失败: {e}</p>"
    

    
    def train_ctr_model(self):
        """训练CTR模型"""
        try:
            # 获取CTR数据
            ctr_data = self.ctr_collector.export_data()
            
            if not ctr_data['records'] or len(ctr_data['records']) < 5:
                return (
                    "<p style='color: #dc3545;'>❌ CTR数据不足，至少需要5条记录才能训练模型</p>",
                    "<p>请先进行一些搜索和点击操作，收集足够的CTR数据后再训练模型。</p>",
                    "<p>暂无特征权重数据</p>"
                )
            
            # 训练模型
            result = self.ctr_model.train(ctr_data['records'])
            
            if 'error' in result:
                return (
                    f"<p style='color: #dc3545;'>❌ 训练失败: {result['error']}</p>",
                    "<p>请检查数据格式或重试。</p>",
                    "<p>暂无特征权重数据</p>"
                )
            
            # 更新搜索引擎的CTR模型
            self.search_engine.ctr_model = self.ctr_model
            
            # 生成训练结果报告
            status_html = f"""
            <p style='color: #28a745;'>✅ CTR模型训练成功！</p>
            <p><strong>模型已保存并加载到搜索引擎中</strong></p>
            """
            
            result_html = f"""
            <div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <h3>🤖 CTR模型训练结果</h3>
                
                <h4>模型性能指标</h4>
                <ul>
                    <li><strong>AUC</strong>: {result['auc']:.4f}</li>
                    <li><strong>精确率</strong>: {result['precision']:.4f}</li>
                    <li><strong>召回率</strong>: {result['recall']:.4f}</li>
                    <li><strong>F1分数</strong>: {result['f1']:.4f}</li>
                </ul>
                
                <h4>训练数据</h4>
                <ul>
                    <li><strong>训练样本数</strong>: {result['train_samples']}</li>
                    <li><strong>测试样本数</strong>: {result['test_samples']}</li>
                    <li><strong>总样本数</strong>: {ctr_data['total_records']}</li>
                </ul>
                
                <h4>使用说明</h4>
                <p>✅ 模型已自动加载到搜索引擎中</p>
                <p>✅ 后续搜索将使用CTR模型进行排序</p>
                <p>✅ 排序分数现在是预测的点击率（0-1之间）</p>
                
                <h4>CTR特征说明</h4>
                <ul>
                    <li><strong>位置衰减</strong>: 1/(位置+1)，位置越靠前权重越高（最重要）</li>
                    <li><strong>位置特征</strong>: 搜索结果中的绝对位置</li>
                    <li><strong>相似度分数</strong>: 原始TF-IDF分数</li>
                    <li><strong>匹配度</strong>: 查询词在摘要中的匹配比例</li>
                    <li><strong>文档历史CTR</strong>: 文档的历史点击率</li>
                    <li><strong>查询长度</strong>: 查询字符串长度</li>
                    <li><strong>文档长度</strong>: 文档摘要长度</li>
                    <li><strong>摘要长度</strong>: 摘要字符串长度</li>
                    <li><strong>查询历史CTR</strong>: 查询的历史点击率</li>
                </ul>
            </div>
            """
            
            # 生成特征权重可视化
            weights_html = "<p>暂无特征权重数据</p>"
            if 'feature_weights' in result and result['feature_weights']:
                weights = result['feature_weights']
                sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                
                weights_html = """
                <div style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <h3>📊 特征权重排名</h3>
                    <p><em>特征权重绝对值越大，对CTR预测越重要</em></p>
                    <div style="max-height: 300px; overflow-y: auto;">
                """
                
                for feature, weight in sorted_weights:
                    # 创建进度条样式的权重显示
                    percentage = min(weight * 100, 100)  # 限制最大100%
                    weights_html += f"""
                    <div style="margin: 8px 0; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <strong>{feature}</strong>
                            <span>{weight:.4f}</span>
                        </div>
                        <div style="background-color: #f0f0f0; height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="background-color: #007bff; height: 100%; width: {percentage}%; transition: width 0.3s;"></div>
                        </div>
                    </div>
                    """
                
                weights_html += """
                    </div>
                </div>
                """
            
            return status_html, result_html, weights_html
            
        except Exception as e:
            return (
                f"<p style='color: #dc3545;'>❌ 训练失败: {e}</p>",
                "<p>请检查系统配置或重试。</p>",
                "<p>暂无特征权重数据</p>"
            )
    
    def run(self):
        """运行界面"""
        self.interface.launch(share=False, inbrowser=True)

def main():
    """主函数"""
    ui = SearchUI()
    ui.run()

if __name__ == "__main__":
    main() 