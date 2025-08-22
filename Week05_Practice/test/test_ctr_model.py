#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTR模型测试脚本
"""

from online.ctr_model import CTRModel
import numpy as np

def create_test_data():
    """创建测试数据"""
    test_data = []
    
    # 模拟一些CTR数据
    queries = ["人工智能", "机器学习", "深度学习", "神经网络", "自然语言处理"]
    doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    
    for i in range(30):  # 30条记录
        query = queries[i % len(queries)]
        doc_id = doc_ids[i % len(doc_ids)]
        position = (i % 10) + 1  # 位置1-10
        score = 0.5 + np.random.random() * 0.5  # 分数0.5-1.0
        summary = f"这是关于{query}的文档摘要，包含相关技术内容。"
        
        # 模拟点击行为（位置越靠前，点击概率越高）
        click_prob = 0.3 + 0.4 * (1.0 / position)  # 位置衰减
        clicked = 1 if np.random.random() < click_prob else 0
        
        test_data.append({
            'query': query,
            'doc_id': doc_id,
            'position': position,
            'score': score,
            'summary': summary,
            'clicked': clicked
        })
    
    return test_data

def test_ctr_model():
    """测试CTR模型"""
    print("🧪 开始测试CTR模型...")
    
    # 创建测试数据
    test_data = create_test_data()
    print(f"✅ 创建了 {len(test_data)} 条测试数据")
    
    # 统计点击情况
    clicks = sum(1 for item in test_data if item['clicked'] == 1)
    print(f"📊 点击次数: {clicks}/{len(test_data)} ({clicks/len(test_data)*100:.1f}%)")
    
    # 创建并训练模型
    model = CTRModel()
    result = model.train(test_data)
    
    if 'error' in result:
        print(f"❌ 训练失败: {result['error']}")
        return
    
    print("✅ 模型训练成功!")
    print(f"📈 AUC: {result['auc']:.4f}")
    print(f"📈 精确率: {result['precision']:.4f}")
    print(f"📈 召回率: {result['recall']:.4f}")
    print(f"📈 F1分数: {result['f1']:.4f}")
    
    # 测试预测
    print("\n🔮 测试预测功能...")
    test_query = "人工智能"
    test_doc_id = "doc1"
    test_position = 1
    test_score = 0.8
    test_summary = "这是关于人工智能的详细文档，包含机器学习相关内容。"
    
    ctr_score = model.predict_ctr(test_query, test_doc_id, test_position, test_score, test_summary)
    print(f"📊 预测CTR: {ctr_score:.4f}")
    
    # 测试不同位置的CTR
    print("\n📊 不同位置的CTR预测:")
    for pos in [1, 3, 5, 7, 10]:
        ctr = model.predict_ctr(test_query, test_doc_id, pos, test_score, test_summary)
        print(f"   位置 {pos}: {ctr:.4f}")
    
    print("\n✅ CTR模型测试完成!")

if __name__ == "__main__":
    test_ctr_model() 