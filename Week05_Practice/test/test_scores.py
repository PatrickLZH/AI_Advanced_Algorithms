#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试分数计算脚本
"""

from offline.offline_index import InvertedIndex
from online.search_engine import SearchEngine

def test_tfidf_scores():
    """测试TF-IDF分数计算"""
    print("🧪 测试TF-IDF分数计算...")
    
    # 创建索引
    index = InvertedIndex()
    
    # 添加一些测试文档
    test_docs = {
        "doc1": "人工智能是计算机科学的一个分支，它企图了解智能的实质。",
        "doc2": "机器学习是人工智能的一个子集，使用统计学方法让计算机学习。",
        "doc3": "深度学习是机器学习的一个分支，基于人工神经网络。"
    }
    
    for doc_id, content in test_docs.items():
        index.add_document(doc_id, content)
    
    # 测试搜索
    query = "人工智能"
    print(f"\n查询: {query}")
    
    results = index.search(query, top_k=3)
    for doc_id, score, summary in results:
        print(f"  - {doc_id}: {score:.4f}")
        print(f"    摘要: {summary[:50]}...")
    
    # 测试搜索引擎
    print(f"\n🔍 测试搜索引擎...")
    engine = SearchEngine()
    
    # 召回
    doc_ids = engine.retrieve(query, top_k=3)
    print(f"召回文档ID: {doc_ids}")
    
    # 排序
    rank_results = engine.rank(query, doc_ids, top_k=3)
    print(f"\n排序结果:")
    for doc_id, tfidf_score, ctr_score, summary in rank_results:
        if ctr_score is not None:
            print(f"  - {doc_id}: TF-IDF={tfidf_score:.4f}, CTR={ctr_score:.4f}")
        else:
            print(f"  - {doc_id}: TF-IDF={tfidf_score:.4f}")
        print(f"    摘要: {summary[:50]}...")
    
    print("\n✅ 分数计算测试完成!")

if __name__ == "__main__":
    test_tfidf_scores() 