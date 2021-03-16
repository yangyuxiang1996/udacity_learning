import json
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# 读取json文件
train_filename = 'train.json'
train_content = pd.read_json(codecs.open(train_filename, mode='r', encoding='utf-8'))

test_filename = 'test.json'
test_content = pd.read_json(codecs.open(test_filename, mode='r', encoding='utf-8'))

print("菜名一共包含 {} 训练数据 和 {} 测试样例。\n".format(len(train_content), len(test_content)))
if len(train_content) == 39774 and len(test_content) == 9944:
    print('数据成功载入！')
else:
    print("数据载入有问题，请检查文件路径！")

pd.set_option('display.max_colwidth', 120)
print(train_content.head())

# 查看有多少种菜品
categories = np.unique(train_content['cuisine'])
print("一共包含 {} 中菜品，分别是:\n{}".format(len(categories), categories))

# 数据提取
train_ingredients = train_content['ingredients']
train_targets = train_content['cuisine']
test_ingredients = test_content['ingredients']
# test_targets = test_content['cuisine']
print(train_ingredients.head())
print(train_targets.head())

# 统计使用佐料的次数，并赋值到字典sum_ingredients中
sum_ingredients = {}
n = train_ingredients.shape[0]
for i in range(n):
    for ingredient in train_ingredients[i]:
        if ingredient in sum_ingredients:
            sum_ingredients[ingredient] += 1
        else:
            sum_ingredients[ingredient] = 1
# print(sum_ingredients)

# Finally, plot the 10 most used ingredients
# plt.style.use(u'ggplot')
fig = pd.DataFrame(sum_ingredients, index=[0]).transpose()[0].sort_values(ascending=False, inplace=False)[:10].plot(kind='barh')
fig.invert_yaxis()
fig = fig.get_figure()
fig.tight_layout()
fig.show()

# TODO: 统计意大利菜系中佐料出现次数，并赋值到italian_ingredients字典中
italian_ingredients = {}
n = train_ingredients.shape[0]
for i in range(n):
    if train_targets[i] == 'italian':
        for ingredient in train_ingredients[i]:
            if ingredient in italian_ingredients:
                italian_ingredients[ingredient] += 1
            else:
                italian_ingredients[ingredient] = 1
fig = pd.DataFrame(italian_ingredients, index=[0]).transpose()[0].sort_values(ascending=False, inplace=False)[:10].plot(kind='barh')
fig.invert_yaxis()
fig = fig.get_figure()
fig.tight_layout()
fig.show()


# 单词清洗
def text_clean(ingredients):
    # 去除单词的标点符号，只保留a-z，A-Z的单词字符
    ingredients = np.array(ingredients).tolist()
    print("菜品佐料：\n{}".format(ingredients[9]))
    ingredients = [[re.sub('[^A-Za-z]', ' ', word) for word in component] for component in ingredients]
    print("去除标点符号后的结果：\n{}".format(ingredients[9]))

    # 去除单词的单复数，时态，只保留单词的词干
    lemma = WordNetLemmatizer()
    ingredients = [" ".join([ " ".join([lemma.lemmatize(w) for w in words.split(" ")])
                              for words in component]) for component in ingredients]
    print("去除时态和单复数之后的结果：\n{}".format(ingredients[9]))
    return ingredients


print("\n处理训练集...")
train_ingredients = text_clean(train_content['ingredients'])
print("\n处理测试集...")
test_ingredients = text_clean(test_content['ingredients'])


"""
在该步骤中，我们将菜品的佐料转换成数值特征向量。考虑到绝大多数菜中都包含salt, water, sugar, butter等，
采用one-hot的方法提取的向量将不能很好的对菜系作出区分。我们将考虑按照佐料出现的次数对佐料做一定的加权，
即：佐料出现次数越多，佐料的区分性就越低。我们采用的特征为TF-IDF，
相关介绍内容可以参考：TF-IDF与余弦相似性的应用（一）：自动提取关键词。
"""
# 特征提取
# 将佐料转换成特征向量
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1),
                             analyzer='word', max_df=.57, binary=False,
                             token_pattern=r'\w+', sublinear_tf=False)
# 处理训练集
train_tfidf = vectorizer.fit_transform(train_ingredients).todense()
# fit_transform()，先拟合数据，再标准化数据

# 处理测试集
test_tfidf = vectorizer.transform(test_ingredients)
# transform()，标准化数据

train_targets = np.array(train_targets).tolist()
train_targets[:10]

"""
调用train_test_split函数将训练集划分为新的训练集和验证集，便于之后的模型精度观测。
从sklearn.model_selection中导入train_test_split
将train_tfidf和train_targets作为train_test_split的输入变量
设置test_size为0.2，划分出20%的验证集，80%的数据留作新的训练集。
设置random_state随机种子，以确保每一次运行都可以得到相同划分的结果。（随机种子固定，生成的随机序列就是确定的）
"""
np.random.seed(42)

X_train, X_valid, y_train, y_valid = train_test_split(train_tfidf, train_targets, test_size=0.2, random_state=42)

"""
训练模型
从sklearn.linear_model导入LogisticRegression
从sklearn.model_selection导入GridSearchCV, 参数自动搜索，只要把参数输进去，就能给出最优的结果和参数，这个方法适合小数据集。
定义parameters变量：为C参数创造一个字典，它的值是从1至10的数组;
定义classifier变量: 使用导入的LogisticRegression创建一个分类函数;
定义grid变量: 使用导入的GridSearchCV创建一个网格搜索对象；将变量'classifier', 'parameters'作为参数传至这个对象构造函数中；
"""
# 训练模型
parameters = {'C': list(range(1, 11))}
classifier = LogisticRegression()

grid = GridSearchCV(classifier, parameters)
grid = grid.fit(X_train, y_train)

valid_predict = grid.predict(X_valid)
valid_score = accuracy_score(y_valid, valid_predict)

print("验证集上的得分为：{}\n".format(valid_score))

# 模型预测
predictions = grid.predict(test_tfidf)
# predictions_score = accuracy_score(test_targets, predictions)

print("预测的测试集个数为：\n{}".format(len(predictions)))
# print("模型预测的结果：\n{}".format(predictions_score))
test_content['cuisine'] = predictions
print(test_content.head(10))

# 保存结果
submit_frame = pd.read_csv('sample_submission.csv')
result = pd.merge(submit_frame, test_content, on="id", how='left')
result = result.rename(index=str, columns={"cuisine_y": "cuisine"})

test_result_name = "tfidf_cuisine_test.csv"
result[['id', 'cuisine']].to_csv(test_result_name, index=False)




























