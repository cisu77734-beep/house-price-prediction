import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import warnings
import os
warnings.filterwarnings('ignore')




train_path =  "train.csv"
test_path =  "test.csv"


# 读取数据
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# One-Hot编码
ohe = OneHotEncoder(sparse_output=False)
hood_ohe = ohe.fit_transform(df_train[['Neighborhood']])

# PCA降维到1-2维
pca = PCA(n_components=2)
hood_pca = pca.fit_transform(hood_ohe)

df_train['Neighborhood_PCA1'] = hood_pca[:, 0]
df_train['Neighborhood_PCA2'] = hood_pca[:, 1]


cluster_features = [
    'OverallQual','GrLivArea','YearBuilt','Neighborhood_PCA1','Neighborhood_PCA2'
]

# 训练集有效数据
print("各特征缺失值数量：")
print(df_train[cluster_features].isnull().sum())

# 处理缺失值（如果有）
train_data = df_train[cluster_features].copy()
train_data = train_data.dropna()  # 或者用 fillna
# 标准化：仅训练集拟合
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

# 肘部法则+轮廓系数选最优K
sil_score_list = []
k_range = range(2,7)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(train_scaled)
    sil_score = silhouette_score(train_scaled, labels)
    sil_score_list.append(sil_score)
    print(f"K={k} 轮廓系数：{sil_score:.4f}")

best_k = k_range[np.argmax(sil_score_list)]
print(f"\n✅ 最优聚类数 K = {best_k}")


# 训练最终聚类模型
k_final = 4  # 你也可以用 best_k
kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
df_train['Cluster'] = kmeans_final.fit_predict(train_scaled)

# ========== 8. 查看结果 ==========
print(f"\n📊 K={k_final} 聚类结果：")
print("\n样本数量：")
print(df_train['Cluster'].value_counts().sort_index())

print("\n特征均值：")
print(df_train.groupby('Cluster')[cluster_features].mean().round(2))

print("\n各聚类的主要社区：")
for cluster_id in range(k_final):
    top_hoods = df_train[df_train['Cluster'] == cluster_id]['Neighborhood'].value_counts().head(3)
    print(f"\n聚类 {cluster_id}：")
    print(top_hoods)

# ========== 9. 保存模型 ==========
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans_final, 'kmeans.pkl')


df_train.to_csv("train_with_cluster.csv2", index=False)
print("\n✅ 已保存所有文件")