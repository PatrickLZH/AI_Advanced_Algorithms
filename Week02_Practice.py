# from modelscope import snapshot_download
from FlagEmbedding import BGEM3FlagModel
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# 正面和负面的文本例子 
pos_texts = [
 "这部电影的剧情和演技让我完全沉浸其中。",
 "我很喜欢这个故事，情感充沛，打动人心。",
 "看完觉得非常温暖，推荐大家观看。",
 "主角的表演令人惊艳，情节引人入胜。",
 "这是一部难得的佳作，让人感动。"
 ]
neg_texts = [
 "剧情毫无亮点，看得很无聊。",
 "浪费时间，完全没有期待中的好。",
 "表演很尴尬，故事也很俗套。",
 "剧情拖沓，结局让人失望。",
 "看完之后非常后悔，不推荐。"
 ]

texts = pos_texts + neg_texts
labels = [1]*len(pos_texts) + [0]*len(neg_texts)

# 2. 获取 embedding
print("Loading BGE-M3 model...")

# model_dir = snapshot_download('BAAI/bge-m3', cache_dir='D:\\MyProject\\Model\\', revision='master')
# print(model_dir)
model_dir = "D:\\MyProject\\Model\\BAAI\\bge-m3"
model = BGEM3FlagModel(model_dir, use_fp16=True)
embeddings = model.encode(texts)['dense_vecs']

# 3. 训练 Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(embeddings, labels)

test_texts = ["这个电影让我感动得落泪了。", "这个电影的情节乱七八糟。"]
test_embeddings = model.encode(test_texts)['dense_vecs']
rst = clf.predict(test_embeddings)
print(test_texts)
print(rst)

# 4. 获取权重
weights = clf.coef_.flatten()
top_k = 10
top_indices = np.argsort(np.abs(weights))[-top_k:][::-1]

print(top_indices)
print(weights[top_indices])

dimensions_filter = [f'dim{i}' for i in top_indices]
weights_filter = weights[top_indices]

# 创建条形图
fig, ax = plt.subplots()
# 绘制条形图
bars = ax.bar(dimensions_filter, weights_filter, color='blue')
# 添加标题和轴标签
ax.set_title('Top-10 Embedding Dimensions for Sentiment Classification')
ax.set_xlabel('Dimension')
ax.set_ylabel('Weight')
# 画上网格
ax.grid(True, linestyle='--', alpha=0.5)  # linestyle 控制网格线样式，alpha 控制透明度
# 在条形上添加权重值
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'+{abs(yval):.2f}' if yval >= 0 else f'{yval:.2f}', 
            ha='center', va='bottom' if yval > 0 else 'top')
# 旋转 X 轴标签
plt.xticks(rotation=45)
# 显示图形
plt.show()
