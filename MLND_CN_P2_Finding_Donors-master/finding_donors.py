import pandas as pd
import numpy as np
from time import time
import visuals as vs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import fbeta_score, accuracy_score, make_scorer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv('census.csv')
print(data.head())

# TODO：总的记录数
n_records = data.shape[0]

# TODO：被调查者的收入大于$50,000的人数
n_greater_50k = (data['income'] == '>50K').sum()

# TODO：被调查者的收入最多为$50,000的人数
n_at_most_50k = (data['income'] != '>50K').sum()

# TODO：被调查者收入大于$50,000所占的比例
greater_percent = n_greater_50k / n_records

# print the result
print("Total number of recorders : {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))

# split the targets and the features
income_raw = data['income']
features_raw = data.drop('income', axis=1)

vs.distribution(features_raw)

# 对于倾斜的数据使用log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

vs.distribution(features_raw)

# 初始化一个Scaler， 并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(features_raw[numerical])

# 显示一个经过缩放的样例记录
print(features_raw.head(1))

# TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
features = pd.get_dummies(features_raw)

# TODO：将'income_raw'编码成数字值
income = income_raw.replace({'<=50K': 0, '>50K': 1})

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# 移除下面一行的注释以观察编码的特征名字
print(encoded)

# 混洗和切分数据
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size=0.2, random_state=0,
                                                    stratify=income)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                  stratify=y_train)
# 显示切分结果
print("The training set has {} samples.".format(X_train.shape[0]))
print("The validation set has {} samples.".format(X_val.shape[0]))
print("The test set has {} samples.".format(X_test.shape[0]))

# 不能使用scikit-learn，你需要根据公式自己实现相关计算。

y_val_pred = pd.Series(np.ones(y_val.shape[0], dtype=int), index=y_val.index)
# TODO： 计算准确率
accuracy = (y_val_pred == y_val).sum() / y_val.shape[0]

# TODO： 计算查准率 Precision
precision = ((y_val_pred == 1) & (y_val == 1)).sum() / (((y_val_pred == 1) & (y_val == 1)).sum() + ((y_val_pred == 1) & (y_val == 0)).sum())

# TODO： 计算查全率 Recall
recall = ((y_val_pred == 1) & (y_val == 1)).sum() / (((y_val_pred == 1) & (y_val == 1)).sum() + ((y_val_pred == 0) & (y_val == 1)).sum())

# TODO： 使用上面的公式，设置beta=0.5，计算F-score
beta = 0.5
fscore = (1 + beta**2) * (precision * recall) / ((beta**2) * precision + recall)

print("Naive Predictor on validation data: \n \
    Accuracy score : {:.4f} \n \
    Precison score : {:.4f} \n \
    Recall score : {:.4f} \n \
    F-score : {:.4f}".format(accuracy, precision, recall, fscore))


# 创建一个训练和预测的流水线
def train_predict(learner, sample_size, X_train, y_train, X_val, y_val):
    """
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_val: features validation set
       - y_val: income validation set
    """

    results = {}

    # TODO: 使用sample_size大小的训练数据来拟合学习器
    # TODO： Fit the learner to the training data using slicing with 'sample_size'
    start = time()
    learner.fit(X_train[0:sample_size], y_train[0:sample_size])
    end = time()

    # TODO: calculate the training time
    results['train_time'] = end - start

    # TODO: predict the learner
    # TODO: 得到在验证集上的预测值, 然后得到对前300个训练数据的预测结果
    start = time()
    predictions_val = learner.predict(X_val)
    predictions_train = learner.predict(X_train[0:300])
    end = time()
    print(predictions_train)

    # TODO: calculate the testing time
    results['pred_time'] = end - start

    # TODO：计算在最前面的300个训练数据的准确率
    results['acc_train'] = accuracy_score(y_train[0:300], predictions_train)

    # TODO: 计算在验证上的准确率
    results['acc_val'] = accuracy_score(y_val, predictions_val)

    # TODO: 计算在最前面的300个训练数据的F-score
    results['f_train'] = fbeta_score(y_train[0:300], predictions_train, beta=0.5)

    # TODO: 计算在验证集上的F-score
    results['f_val'] = fbeta_score(y_val, predictions_val, beta=0.5)

    # 成功
    print("{} trained {} samples.".format(learner.__class__.__name__, sample_size))

    return results


# 初始模型的评估
clf_A = DecisionTreeClassifier(random_state=10)
clf_B = RandomForestClassifier(n_estimators=100, random_state=10)
clf_C = SVC(kernel='rbf', gamma=0.01, C=10)

# TODO：计算1%， 10%， 100%的训练数据分别对应多少点
samples_1 = int(0.01*len(y_train))
samples_10 = int(0.1*len(y_train))
samples_100 = len(y_train)

# 收集学习器的结果
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_val, y_val)

vs.evaluate(results, accuracy, fscore)

# TODO: 选择随机森林为学习模型
# TODO：初始化分类器
random_forest = RandomForestClassifier(random_state=10, n_jobs=-1)

param_grid = { "criterion": ["gini", "entropy"],
               "min_samples_leaf": [1, 5, 10, 25, 50, 70],
               "min_samples_split": [2, 4, 10, 12, 16, 18, 25, 35],
               "n_estimators": [100, 400, 700, 1000, 1500]}

scorer = make_scorer(fbeta_score, beta=0.5)

cv = KFold(n_splits=10, random_state=10)

grid_obj = GridSearchCV(estimator=random_forest, param_grid=param_grid, scoring=scorer, cv=cv)

grid_obj = grid_obj.fit(X_train, y_train)

best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_val)
# 使用最优模型进行预测
best_predictions = best_clf.predict(X_val)

# 打印调优后的模型
print("best model report:\n")
print(best_clf)

# 汇报调优前的分数
print("\nUnoptimized model \n----------")
print("Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions)))
print("F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta=0.5)))

# 汇报调优后的分数
print("\nOptimized model \n----------")
print("Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, best_predictions)))
print("F-score on validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta=0.5)))

