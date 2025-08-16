from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BertSentenceEncoder:
    def __init__(self, model_name='bert-base-uncased'):
        """
        初始化BERT句子编码器
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # 设置为评估模式
    
    def encode_sentences(self, sentences):
        """
        对句子进行编码，提取句子级别的向量表示
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # 对句子进行tokenize
        encoded_inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 使用BERT模型获取隐藏状态
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            # outputs.last_hidden_state: [batch_size, sequence_length, hidden_size]
            last_hidden_states = outputs.last_hidden_state
        
        # 提取句子向量表示的几种方法
        sentence_embeddings = {}
        
        # 方法1: 使用[CLS]标记的向量
        cls_embeddings = last_hidden_states[:, 0, :].numpy()
        sentence_embeddings['cls'] = cls_embeddings
        
        # 方法2: 对所有token的向量取平均
        mean_embeddings = torch.mean(last_hidden_states, dim=1).numpy()
        sentence_embeddings['mean'] = mean_embeddings
        
        # 方法3: 对除了[CLS]和[SEP]外的所有token取平均
        attention_mask = encoded_inputs['attention_mask']
        masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        mean_pooled_embeddings = sum_embeddings / torch.sum(attention_mask, dim=1, keepdim=True)
        sentence_embeddings['mean_pooled'] = mean_pooled_embeddings.numpy()
        
        return sentence_embeddings
    
    def calculate_similarity(self, sentences):
        """
        计算句子间的余弦相似度
        """
        embeddings = self.encode_sentences(sentences)
        
        # 使用不同方法计算相似度
        similarities = {}
        for method, emb in embeddings.items():
            sim_matrix = cosine_similarity(emb)
            similarities[method] = sim_matrix
        
        return similarities

# 展示BERT对上下文语义的理解能力
def demonstrate_context_understanding():
    encoder = BertSentenceEncoder('bert-base-uncased')
    
    # 同一个词在不同语境下的表示差异
    contextual_sentences = [
        "I went to the bank to deposit money.",
        "We sat by the river bank enjoying the view.",
        "The bank approved my loan application."
    ]
    
    print("上下文语义理解能力展示:")
    print("=" * 50)
    print("测试句子:")
    for i, sent in enumerate(contextual_sentences):
        print(f"{i+1}. {sent}")
    
    print("\n'bank'在不同语境下的语义表示:")
    
    # 提取每个句子中'bank'的向量表示
    embeddings = encoder.encode_sentences(contextual_sentences)
    
    for method, emb in embeddings.items():
        print(f"\n{method.upper()}方法:")
        print(f"  句子1 (金融机构) 向量范数: {np.linalg.norm(emb[0]):.4f}")
        print(f"  句子2 (河岸) 向量范数: {np.linalg.norm(emb[1]):.4f}")
        print(f"  句子3 (银行) 向量范数: {np.linalg.norm(emb[2]):.4f}")
        print(f"  句子1与句子2相似度: {cosine_similarity([emb[0]], [emb[1]])[0][0]:.4f}")
        print(f"  句子1与句子3相似度: {cosine_similarity([emb[0]], [emb[2]])[0][0]:.4f}")


if __name__ == '__main__':
    # 初始化BERT编码器
    encoder = BertSentenceEncoder('bert-base-uncased')

    # 示例句子
    sentences = [
        "The cat sits on the mat.",
        "A cat is sitting on a rug.",
        "Dogs are playing in the park.",
        "Machine learning is fascinating.",
        "Artificial intelligence is amazing."
    ]

    print("句子向量编码示例:")
    print("=" * 50)

    # 编码句子
    embeddings = encoder.encode_sentences(sentences)

    # 展示不同方法的向量维度
    for method, emb in embeddings.items():
        print(f"{method.upper()}方法:")
        print(f"  向量形状: {emb.shape}")
        print(f"  向量示例 (前10维): {emb[0][:10]}")
        print()

    # 计算句子相似度
    print("句子相似度计算:")
    print("=" * 50)

    similarities = encoder.calculate_similarity(sentences)

    for method, sim_matrix in similarities.items():
        print(f"\n{method.upper()}方法相似度矩阵:")
        print(f"句子对\t\t\t\t\t\t\t\t\t相似度")
        print("-" * 100)
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences):
                if i < j:  # 只显示上三角矩阵
                    print(f"'{sentences[i]}' vs '{sentences[j]}'\t{sim_matrix[i][j]:.4f}\n")

    # 运行演示
    demonstrate_context_understanding()