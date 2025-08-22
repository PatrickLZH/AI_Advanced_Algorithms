#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索引擎实现 - 实现trigger和ranker方法
负责召回+排序的核心功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from online.search_interface import SearchInterface
from offline.offline_index import InvertedIndex
from online.ctr_model import CTRModel
from typing import List, Dict, Any, Tuple
import os
import math

class SearchEngine(SearchInterface):
    """搜索引擎实现类"""
    
    def __init__(self):
        self.index = None
        self.current_results = []  # 当前搜索结果
        self.ctr_model = CTRModel()  # CTR模型
        self.load_index()
        self.load_ctr_model()
    
    def load_index(self):
        """加载索引"""
        try:
            # 从离线模块加载索引
            self.index = InvertedIndex()
            
            # 检查是否有保存的索引文件
            if os.path.exists('models/index_data.json'):
                self.index.load_from_file('models/index_data.json')
                print("✅ 从文件加载索引成功")
            else:
                # 创建示例索引
                from offline.offline_index import create_sample_documents
                documents = create_sample_documents()
                for doc_id, content in documents.items():
                    self.index.add_document(doc_id, content)
                print("✅ 创建示例索引成功")
            
            print(f"   索引统计: {self.index.get_index_stats()}")
            
        except Exception as e:
            print(f"❌ 加载索引失败: {e}")
            self.index = None
    
    def load_ctr_model(self):
        """加载CTR模型"""
        try:
            if self.ctr_model.load_model():
                print("✅ CTR模型加载成功")
            else:
                print("⚠️  未找到CTR模型文件，将使用默认排序")
        except Exception as e:
            print(f"❌ 加载CTR模型失败: {e}")
    
    def retrieve(self, query: str, top_k: int = 20) -> List[str]:
        """召回阶段：返回初步相关的文档ID列表（按TF-IDF分数粗排）"""
        if not self.index:
            raise Exception("索引未加载")
        if not query.strip():
            return []
        # 只返回文档ID，按分数降序
        results = self.index.search(query.strip(), top_k=top_k)
        doc_ids = [doc_id for doc_id, score, summary in results]
        return doc_ids
    
    def rank(self, query: str, doc_ids: List[str], top_k: int = 10) -> List[Tuple[str, float, str]]:
        """排序阶段：对召回的文档ID进行精排，使用CTR模型重新排序"""
        if not self.index:
            raise Exception("索引未加载")
        if not query.strip() or not doc_ids:
            return []
        
        # 第一步：使用离线索引的search方法获取TF-IDF分数和摘要
        # 这样可以确保分数计算的一致性
        full_results = self.index.search(query.strip(), top_k=len(doc_ids))
        
        # 过滤出在doc_ids中的结果
        filtered_results = []
        for doc_id, score, summary in full_results:
            if doc_id in doc_ids:
                filtered_results.append((doc_id, score, summary))
        
        if not filtered_results:
            return []
        
        # 第二步：使用CTR模型重新排序
        if self.ctr_model.is_trained:
            # 有CTR模型时，计算CTR分数并重新排序
            ctr_scores = {}
            for position, (doc_id, tfidf_score, summary) in enumerate(filtered_results, 1):
                ctr_score = self.ctr_model.predict_ctr(query, doc_id, position, tfidf_score, summary)
                ctr_scores[doc_id] = (ctr_score, tfidf_score, summary)
            
            # 按CTR分数排序
            sorted_results = sorted(ctr_scores.items(), key=lambda x: x[1][0], reverse=True)
            
            # 构建最终结果（同时保存TF-IDF和CTR分数）
            results = []
            for doc_id, (ctr_score, tfidf_score, summary) in sorted_results[:top_k]:
                # 返回元组：(doc_id, tfidf_score, ctr_score, summary)
                results.append((doc_id, tfidf_score, ctr_score, summary))
        else:
            # 没有CTR模型时，使用TF-IDF分数排序
            sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)
            
            # 构建最终结果（只返回TF-IDF分数）
            results = []
            for doc_id, tfidf_score, summary in sorted_results[:top_k]:
                # 返回元组：(doc_id, tfidf_score, None, summary)
                results.append((doc_id, tfidf_score, None, summary))
        
        self.current_results = results
        return results
    
    def get_document(self, doc_id: str) -> str:
        """获取文档内容"""
        if not self.index:
            raise Exception("索引未加载")
        
        return self.index.get_document(doc_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if not self.index:
            return {
                'total_documents': 0,
                'total_terms': 0,
                'average_doc_length': 0
            }
        
        return self.index.get_index_stats()
    
    def get_current_results(self) -> List[Tuple[str, float, float, str]]:
        """获取当前搜索结果"""
        return self.current_results 