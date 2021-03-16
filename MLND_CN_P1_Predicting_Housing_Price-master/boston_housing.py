import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import visuals as vs
from scipy.special import boxcox1p
import seaborn as sns
import warnings
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


plt.style.use('seaborn')  # use seaborn style 使用seaborn风格
warnings.filterwarnings('ignore')

print("你已成功载入所有库！")

'''
第一步：导入数据
'''
# 1 TODO：载入波士顿房屋的数据集：使用pandas载入csv，并赋值到data_df
data_df = pd.read_csv('housedata.csv')
# 成功载入的话输出训练数据行列数目
print("Boston housing data has {} data points with {} variables each.".format(*data_df.shape))


'''
第二步：数据分析
'''
# 2.1 TODO: 打印出前7条data_df
print(data_df.head(7))

# 2.2 TODO: 删除data_df中的Id特征（保持数据仍在data_df中，不更改变量名）
data_df.drop(columns=['Id'], inplace=True)

# 2.3 TODO: 使用describe方法观察data_df各个特征的统计信息
# data_features = data_df.drop(columns=['SalePrice'])
# data_target = data_df['SalePrice']
data_df.describe(include='all')

# 使用matplotlib库中的scatter方法 绘制'GrLivArea'和'SalePrice'的散点图，x轴为'GrLivArea'，y轴为'SalePrice'，观察数据
plt.scatter(data_df['GrLivArea'], data_df['SalePrice'], c='blue', marker='.', s=50)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()

# 3.1.2
# TODO:从data_df中删除 GrLivArea大于4000 且 SalePrice低于300000 的值
index = data_df[(data_df['GrLivArea'] > 4000) & (data_df['SalePrice'] < 300000)].index
data_df.drop(index=index, inplace=True)

# TODO:重新绘制GrLivArea和SalePrice的关系图，确认异常值已删除
plt.scatter(data_df['GrLivArea'], data_df['SalePrice'], c='blue', marker='.', s=50)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()

# 3.2.1 TODO 统计并打印出超过25%的空数据的特征，你可以考虑使用isna()
limit_percent = 0.25
limit_value = len(data_df)*limit_percent
print(list(data_df.columns[data_df.isna().sum() > limit_value]))

# 3.2.2 TODO 根据data_description.txt特征描述,使用fillna方法填充空数据
# 确定所有空特征
missing_columns = list(data_df.columns[data_df.isna().sum() != 0])
# 确定哪些是类别特征，哪些是数值特征
missing_numerical = list(data_df[missing_columns].dtypes[data_df[missing_columns].dtypes != 'object'].index)
missing_category = [i for i in missing_columns if i not in missing_numerical]
print("missing_numerical: ", missing_numerical)
print("missing_category: ", missing_category)

# 需要填充众数的特征
fill_Mode = ['Electrical']
# 需要填充None的特征
fill_None = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
             'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
             'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
# 需要填充0的特征
fill_0 = ['GarageYrBlt']
# 需要填充中位数的特征
fill_median = ['LotFrontage', 'MasVnrArea']

# 填充众数：
mode = data_df[fill_Mode].mode().iloc[0,0]
data_df.fillna(value={fill_Mode[0]: mode}, inplace=True)
# 填充None：
d_None = {}
for i in fill_None:
    d_None[i] = 'None'
data_df.fillna(value=d_None, inplace=True)
# 填充0：
data_df[fill_0].fillna(value=0, inplace=True)
# 填充中位数：
data_df[fill_median].fillna(data_df[fill_median].median(), inplace=True)

# TODO 问题4.1：绘制'SalePrice'的直方图，并说明该直方图属于什么分布
plt.hist(data_df['SalePrice'], bins=100, normed=False, color=None)
plt.xlabel('SalePrice')
plt.show()

corrmat = data_df.corr().abs()  # 计算连续型特征之间的相关系数
# 将于SalePrice的相关系数大于5的特征取出来，并按照SalePrice降序排列，然后取出对应的特征名，保存在列表中
top_corr = corrmat[corrmat["SalePrice"]>0.5].sort_values(by = ["SalePrice"], ascending = False).index
cm = abs(np.corrcoef(data_df[top_corr].values.T))  # 注意这里要转置，否则变成样本之间的相关系数，而我们要计算的是特征之间的相关系数
f, ax = plt.subplots(figsize=(20, 9))
sns.set(font_scale=1.3)
hm = sns.heatmap(cm, cbar=True, annot=True,
                 square=True, fmt='.2f', annot_kws={'size': 13},
                 yticklabels=top_corr.values, xticklabels=top_corr.values);
data_df = data_df[top_corr]

data_df['SalePrice'] = np.log1p(data_df['SalePrice'])
numeric_features = list(data_df.columns)
numeric_features.remove('SalePrice')
for feature in numeric_features:
    data_df[feature] = boxcox1p(data_df[feature], 0.15)

scaler = StandardScaler()
scaler.fit(data_df[numeric_features])
data_df[numeric_features] = scaler.transform(data_df[numeric_features])

'''
第三步：建立模型
'''


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
            true and predicted values based on the metric chosen. """
    score = r2_score(y_true, y_predict)
    return score


# TODO: 6.1 将data_df分割为特征和目标变量
labels = data_df['SalePrice']  # TODO：提取SalePrice作为labels
features = data_df.drop(['SalePrice'], axis=1)  # TODO：提取除了SalePrice以外的特征赋值为features

# TODO: 打乱并分割训练集与测试集
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print("Training and testing split was successful.")

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, labels)
plt.show()

vs.ModelComplexity(X_train, y_train)
plt.show()


def fit_model(X, y):
    cross_validator = KFold(10, random_state=42)
    regressor = DecisionTreeRegressor(random_state=42)
    params = {'max_depth': list(range(1, 11))}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, params, scoring_fnc, cv=cross_validator)
    grid.fit(X, y)

    return grid.best_estimator_


reg = fit_model(X_train, y_train)
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

depth = reg.get_params()['max_depth']
regressor = DecisionTreeRegressor(max_depth=depth)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
score = performance_metric(y_test, y_pred)
print("the R2 Score is ", score)











