import jieba
import os
from gensim.models import word2vec

def load_stopwords(stopwords_path):
    with open(stopwords_path,'r',encoding='utf-8') as f:
        stopwords = set(f.read().split('\n'))
        return stopwords

# 字词分割，对整个文件内容进行字词分割
def segment_lines(input_path,stopwords):
    stopwords = load_stopwords(stopwords_path)
    with open(input_path, 'r', encoding='utf-8') as f:
        document = f.read()
        document_cut = jieba.cut(document)
        sentence_segment=[]
        for word in document_cut:
            if word not in stopwords:
                sentence_segment.append(word)
        result = ' '.join(sentence_segment)
        result = result.encode('utf-8')
        with open(output_path, 'wb') as f2:
            f2.write(result)

stopwords_path = os.path.abspath('./txt/stopwords.txt')
# print(f'stopwords_path:{stopwords_path}')
input_path = os.path.abspath('./txt/source/three_kingdoms.txt')
# print(f'input_path:{input_path}')
output_path = os.path.abspath('./txt/segment/three_kingdoms_segment.txt')
# print(f'output_path:{output_path}')
segment_lines(input_path,stopwords_path)

# 如果目录中有多个文件，可以使用PathLineSentences
segment_folder = os.path.abspath('./txt/segment')
# print(f'segment_folder:{segment_folder}')
sentences = word2vec.PathLineSentences(segment_folder)

# 调整后的模型参数
model = word2vec.Word2Vec(
    sentences, 
    vector_size=128,     # 增加向量维度
    window=7,           # 增大窗口大小
    min_count=1,         # 降低最小词频
    epochs=15,           # 增加训练轮数
    sg=1,                 # 使用skip-gram模型（通常对中文效果更好）
    seed=42,              # 固定随机种子
    workers=1             # 使用单线程以确保完全可重现
)
# model.save('Word2Vec.model')
# model = word2vec.Word2Vec.load('word2Vec.model')
print(model.wv.similarity('曹操', '吴国太'))
print(model.wv.similarity('司马师', '司马懿'))
print(model.wv.most_similar(positive=['刘禅', '司马懿'], negative=['刘备'])[0])