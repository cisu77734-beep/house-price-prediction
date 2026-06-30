# House Price Prediction with K-Means & XGBoost
房价预测思路：
数据集一共有79个特征；

一，进行相关性分析，获得与目标特征【房价】相关性最高的10个特征，如图

二，同时进行探索性数据分析，
1，先画图分析目标变量【房价】，如图

图一，从偏度看，房价右偏（偏度: 1.8829），存在高房价离群点，对数变换后，趋近正态分布，建模时应对房价进行对数变换。
2,对数据进行缺失值处理
分为数值型特征和分类型特征
   - 数值型特征（GarageArea, TotalBsmtSF 等）：缺失值用 0 填充（表示没有）
   - 分类型特征（BsmtQual, FireplaceQu 等）：缺失值用 'None' 填充
   - 或使用均值/中位数填充
3，核心特征分析：【OveralQual】，【GrLivArea】

可以看出，【OverallQual】越高，房价越高，呈明显正相关。

将GrliveArea与SalePrice做散点图，发现存在面积大但价格低的离群点，这样的点做现实发现具有实际价值，保留，对GrliveArea做对数变换。

TotalBsmtSF同样处理；

年份特征分析：
   - YearBuilt：新建房屋房价更高，但趋势不够平滑
   - HouseAge（房龄）：房龄越低（房子越新），房价越高
建模时使用 HouseAge 替代 YearBuilt，与房价的负相关关系更直接、稳定。

组合特征【 AreaPerRoom 】 '平均每间房面积' :
   GrLivArea (原始) 相关性: 0.709
   TotRmsAbvGrd (原始) 相关性: 0.534
   AreaPerRoom (组合) 相关性: 0.541
   ✅ 建模建议: 在建模时加入组合特征 `AreaPerRoom`，它能反映居住空间的舒适度。
三，K-means聚类
 使用K-Means聚类，选用三个数值型特征['OverallQual', 'GrLivArea',  'YearBuilt']，和一个类别型特征 ['Neighborhood']进行聚类，K=4.
 独热编码：将 ['Neighborhood']转换为数值型，其中有20个社区，所以使用PCB降维，
最终降维压缩成两个新特征 ['Neighborhood1'] ['Neighborhood2'].
聚类结果生成四个组别标签【Cluster】
解释：
Cluster 0 ：老旧低质小户型房源
整体质量5.18 最低 
居住面积1306.95 最小 
建造年份1946.66 最早、房龄最老
特征：房龄老旧、房屋品质差、户型面积局促，属于老旧低端刚需住宅。 
Cluster 1 ：中等品质次新普通住宅
整体质量 6.64 中等偏上 
居住面积 1480.50 适中 
建造年份1997.89 最新，房龄年轻
特征：房龄新、品质中等、面积常规，属于次新普通自住型住宅。 
Cluster 2 ：高品质大户型改善房源
整体质量7.30 全局最高 
居住面积1850.28 远高于其他类别，面积最大 
建造年份 1996.06，房龄较新
特征：品质最优、户型面积最大、房龄较新，是典型中高端大户型改善住宅。 
Cluster 3 ：低质偏老旧常规户型房源
整体质量 5.36 略高于 0 类、仍偏低 
居住面积 1310.31 和 0 类几乎持平，小户型水平 
建造年份 1960.00，房龄偏老但比 0 类新
特征：品质偏低、面积偏小、房龄中等偏老，介于 0 类和 1 类之间，属于偏老旧刚需过渡型住宅。 
四类在建筑品质、户型面积、建造年代上呈现明显梯度差异，聚类划分具备良好业务可解释性，可作为衍生特征用于房价回归建模。

四，最终建模特征 ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'GarageArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearRemodAdd', 'Neighborhood', 'KitchenQual', 'BsmtQual', 'FireplaceQu', 'MSZoning', 'HouseAge', 'AreaPerRoom', 'Cluster_Label'] #KMeans 生成的 Cluster_Label (4个类别) 作为新的输入特征。
五，建模
使用线性回归作为基准模型，xgboost作为实验模型
将以上特征作为输入值，完成建模











