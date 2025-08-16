import pandas as pd
import random
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 练习一：用户画像的模拟与生成
random.seed(42)
N = 1000
df = pd.DataFrame(columns=['user_id', 'sex', 'city', 'consumption_level', 'age', 'recent_active_days'])
df['user_id'] = [str(i).zfill(4) for i in range(N)]
df['sex'] = random.choices(['男','女','未透露'],[0.4,0.4,0.2],k=N)
# df['sex'] = random.choices(['男','女'],[0.5,0.5],k=N)
# print(df['sex'].groupby(df['sex']).count())
df['city'] = random.choices(['北京','上海','广州','深圳','其他'],[0.2,0.2,0.2,0.2,0.2],k=N)
# print(df['city'].groupby(df['city']).count())
df['consumption_level'] = random.choices(['高','中','低'],[0.3,0.4,0.3],k=N)
# print(df['consumption_level'].groupby(df['consumption_level']).count())
df['age'] = [int(random.gauss(mu=30,sigma=5)) for _ in range(N)] 
# print(df['age'].groupby(df['age']).count())
df['recent_active_days'] = [int(random.expovariate(0.1))+1 for _ in range(N)]
# print(df['recent_active_days'].groupby(df['recent_active_days']).count())
# df.to_csv('user_info.csv',index=False,encoding='utf-8-sig')

# 练习二：简单留出法体验
train,test = train_test_split(df,test_size=0.2,random_state=42,shuffle=True,stratify=None)
# print(train.shape,test.shape)
print(train['consumption_level'].groupby(train['consumption_level']).count())
print(test['consumption_level'].groupby(test['consumption_level']).count())

# 练习三：分层抽样体验
train,test = train_test_split(df,test_size=0.2,random_state=42,shuffle=True,stratify=df['consumption_level'])
# print(train.shape,test.shape)
print(train['consumption_level'].groupby(train['consumption_level']).count())
print(test['consumption_level'].groupby(test['consumption_level']).count())

# 练习四：将你的用户数据“向量化”
# from modelscope import snapshot_download
# model_dir = snapshot_download('BAAI/bge-m3', cache_dir='D:\\MyProject\\Model\\', revision='master')
# print(model_dir)

from FlagEmbedding import BGEM3FlagModel
model_dir = 'D:\\MyProject\\Model\\BAAI\\bge-m3'
model = BGEM3FlagModel(model_dir, use_fp16=True)

words = df['sex'].unique().tolist() + df['city'].unique().tolist() + df['consumption_level'].unique().tolist()
# print(words)

words_embed = {word:model.encode(word)['dense_vecs'] for word in words}
# print(words_embed)

sex_embed = np.stack(df['sex'].map(words_embed).values)
city_embed = np.stack(df['city'].map(words_embed).values)
consumption_level_embed = np.stack(df['consumption_level'].map(words_embed).values)
age_embed = np.expand_dims(df['age'].values, axis=1)
recent_active_days_embed = np.expand_dims(df['recent_active_days'].values, axis=1)

X = np.concat([sex_embed,city_embed,consumption_level_embed,age_embed,recent_active_days_embed], axis=1)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# ************************************
kmeans = KMeans(n_clusters=9)
kmeans.fit(X_std)
df['kmeans'] = kmeans.labels_.tolist()

# --- 3. PCA三维降维 ---
CAT_COLS = ["sex", "city", "consumption_level"]
NUM_COLS = ["age", "recent_active_days"]
COLOR_COL = ["kmeans"]

pca = PCA(n_components=3)
user_3d = pca.fit_transform(X_std)
exp_var_3d = pca.explained_variance_ratio_
print(f"\n--- PCA降至3维 ---")
for i, var in enumerate(exp_var_3d, 1):
    print(f"主成分{i}解释的方差：{var:.2%}")
print(f"累计解释的方差：{np.sum(exp_var_3d):.2%}")

# --- 4. 3D静态可视化（Matplotlib） ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
def plot_3d_matplotlib(X_3d, df, exp_var, color_cols, n_label=20):
    for color_col in color_cols:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        color_map = {
            k: c for k, c in zip(sorted(df[color_col].unique()), plt.cm.tab10.colors)
        }
        colors = df[color_col].map(color_map)
        ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=colors, alpha=0.5, s=40)
        # 随机点标签
        idxs = random.sample(range(len(df)), min(n_label, len(df)))
        for i in idxs:
            label = "-".join(str(df.loc[i, col]) for col in CAT_COLS + NUM_COLS)
            ax.text(X_3d[i, 0], X_3d[i, 1], X_3d[i, 2], label, fontsize=7, alpha=0.7)
        # 图例
        for level, color in color_map.items():
            ax.scatter([], [], [], color=color, label=level)
        ax.set_title("用户画像PCA三维可视化", fontsize=15)
        ax.set_xlabel(f"主成分1 ({exp_var[0]:.1%})")
        ax.set_ylabel(f"主成分2 ({exp_var[1]:.1%})")
        ax.set_zlabel(f"主成分3 ({exp_var[2]:.1%})")
        ax.legend(title=color_col)
        plt.tight_layout()
        plt.show()

plot_3d_matplotlib(user_3d, df, exp_var_3d, COLOR_COL, n_label=20)

# ************************************
# pca = PCA(n_components=2)
# X_std_pca = pca.fit_transform(X_std)
# df['embed_pca'] = X_std_pca.tolist()

# df[['pca1', 'pca2']] = pd.DataFrame(df['embed_pca'].tolist(), index=df.index)

# # 设置 matplotlib 字体为 SimHei 显示中文
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
# plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# for var in ['sex', 'city', 'consumption_level']:
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(data=df, x='pca1', y='pca2', hue=var, palette='Set2', alpha=0.8)
#     plt.title(f'User Embedding Visualization with PCA ({var})')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
