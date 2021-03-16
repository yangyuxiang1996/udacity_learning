
# 机器学习纳米学位
## 非监督学习
## 项目 3: 创建用户分类

## 开始

在这个项目中，你将分析一个数据集的内在结构，这个数据集包含很多客户真对不同类型产品的年度采购额（用**金额**表示）。这个项目的任务之一是如何最好地描述一个批发商不同种类顾客之间的差异。这样做将能够使得批发商能够更好的组织他们的物流服务以满足每个客户的需求。

这个项目的数据集能够在[UCI机器学习信息库](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)中找到.因为这个项目的目的，分析将不会包括 'Channel' 和 'Region' 这两个特征——重点集中在6个记录的客户购买的产品类别上。

运行下面的的代码单元以载入整个客户数据集和一些这个项目需要的 Python 库。如果你的数据集载入成功，你将看到后面输出数据集的大小。


```python
# 检查python版本
from sys import version_info
if version_info.major != 3:
    raise Exception('请使用Python 3.x 来完成此项目')
```


```python
# 引入项目所需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visuals as vs
from IPython.display import display # 使得我们可以对DataFrame使用display()函数

# 设置以内联的形式显示matplotlib绘制的图片（在notebook中显示更美观）
%matplotlib inline
# 高分辨率显示
# %config InlineBackend.figure_format='retina'

# 载入整个客户数据集
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis=1, inplace=True)
    print("Whole sale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")
```

    Whole sale customers dataset has 440 samples with 6 features each.
    


```python
display(data.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12669</td>
      <td>9656</td>
      <td>7561</td>
      <td>214</td>
      <td>2674</td>
      <td>1338</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7057</td>
      <td>9810</td>
      <td>9568</td>
      <td>1762</td>
      <td>3293</td>
      <td>1776</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6353</td>
      <td>8808</td>
      <td>7684</td>
      <td>2405</td>
      <td>3516</td>
      <td>7844</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13265</td>
      <td>1196</td>
      <td>4221</td>
      <td>6404</td>
      <td>507</td>
      <td>1788</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22615</td>
      <td>5410</td>
      <td>7198</td>
      <td>3915</td>
      <td>1777</td>
      <td>5185</td>
    </tr>
  </tbody>
</table>
</div>


## 分析数据
在这部分，你将开始分析数据，通过可视化和代码来理解每一个特征和其他特征的联系。你会看到关于数据集的统计描述，考虑每一个属性的相关性，然后从数据集中选择若干个样本数据点，你将在整个项目中一直跟踪研究这几个数据点。

运行下面的代码单元给出数据集的一个统计描述。注意这个数据集包含了6个重要的产品类型：**'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**和 **'Delicatessen'**。想一下这里每一个类型代表你会购买什么样的产品。


```python
# TODO: 描述数据集
display(data.describe())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12000.297727</td>
      <td>5796.265909</td>
      <td>7951.277273</td>
      <td>3071.931818</td>
      <td>2881.493182</td>
      <td>1524.870455</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12647.328865</td>
      <td>7380.377175</td>
      <td>9503.162829</td>
      <td>4854.673333</td>
      <td>4767.854448</td>
      <td>2820.105937</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>55.000000</td>
      <td>3.000000</td>
      <td>25.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3127.750000</td>
      <td>1533.000000</td>
      <td>2153.000000</td>
      <td>742.250000</td>
      <td>256.750000</td>
      <td>408.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8504.000000</td>
      <td>3627.000000</td>
      <td>4755.500000</td>
      <td>1526.000000</td>
      <td>816.500000</td>
      <td>965.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16933.750000</td>
      <td>7190.250000</td>
      <td>10655.750000</td>
      <td>3554.250000</td>
      <td>3922.000000</td>
      <td>1820.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>112151.000000</td>
      <td>73498.000000</td>
      <td>92780.000000</td>
      <td>60869.000000</td>
      <td>40827.000000</td>
      <td>47943.000000</td>
    </tr>
  </tbody>
</table>
</div>


### 练习: 选择样本
为了对客户有一个更好的了解，并且了解代表他们的数据将会在这个分析过程中如何变换。最好是选择几个样本数据点，并且更为详细地分析它们。在下面的代码单元中，选择**三个**索引加入到索引列表`indices`中，这三个索引代表你要追踪的客户。我们建议你不断尝试，直到找到三个明显不同的客户。


```python
# TODO: 从数据集中选择三个你希望抽样的数据点的索引
indices = [47, 86, 181]

# 为选择的样本创建一个DataFrame
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop=True)
print("Chosen samples of whole sale customers dataset:")
display(samples)
```

    Chosen samples of whole sale customers dataset:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>44466</td>
      <td>54259</td>
      <td>55571</td>
      <td>7782</td>
      <td>24171</td>
      <td>6465</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22925</td>
      <td>73498</td>
      <td>32114</td>
      <td>987</td>
      <td>20070</td>
      <td>903</td>
    </tr>
    <tr>
      <th>2</th>
      <td>112151</td>
      <td>29627</td>
      <td>18148</td>
      <td>16745</td>
      <td>4948</td>
      <td>8550</td>
    </tr>
  </tbody>
</table>
</div>



```python
import seaborn as sns
%config InlineBackend.figure_format='retina'
samples_bar = samples.append(data.describe().loc[['50%', 'mean']])
samples_bar.plot(kind='bar', figsize=(14, 6))
```

    <matplotlib.axes._subplots.AxesSubplot at 0x1fc61493ac8>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512141216407.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)

### 问题 1
在你看来你选择的这三个样本点分别代表什么类型的企业（客户）？对每一个你选择的样本客户，通过它在每一种产品类型上的花费与数据集的统计描述进行比较，给出你做上述判断的理由。


**提示：** 企业的类型包括超市、咖啡馆、零售商以及其他。注意不要使用具体企业的名字，比如说在描述一个餐饮业客户时，你不能使用麦当劳。你能使用各个平均值作为参考来比较样本，平均值如下

* Fresh: 12000.2977
* Milk: 5796.2
* Grocery: 7951.3
* Detergents_paper: 2881.4
* Delicatessen: 1524.8

了解这一点后，你应该如何比较呢？这对推动你了解他们是什么类型的企业有帮助吗？

**回答:**
1. 第一个客户属于大型超市，因为其对于生鲜牛奶杂货洗涤剂的需求远远大于该产品的平均值和中位数，这属于大型超市的特点；
2. 第二个客户属于咖啡饮料行业，因为其对于牛奶的需求非常高，同时对于生鲜杂货的需求也不少；
3. 第三个客户属于餐馆行业，因为对于生鲜的需求巨大，同时牛奶杂货洗涤剂熟食品的需求也高于平均值，这比较符合餐馆行业的特点

**review：**

和数据集的统计特征进行了很好的比较。同时，我们建议选择 percentile 而非 mean，因为在未知数据分布的情况下使用均值作为比较对象是比较危险的——因为不清楚概率分布，所以用 percentile、median 这样的统计特征会相对好一点。更多你可以参考描述统计学、或者数据的统计量特征。


### 练习: 特征相关性
一个有趣的想法是，考虑这六个类别中的一个（或者多个）产品类别，是否对于理解客户的购买行为具有实际的相关性。也就是说，当用户购买了一定数量的某一类产品，我们是否能够确定他们必然会成比例地购买另一种类的产品。有一个简单的方法可以检测相关性：我们用移除了某一个特征之后的数据集来构建一个监督学习（回归）模型，然后用这个模型去预测那个被移除的特征，再对这个预测结果进行评分，看看预测结果如何。

在下面的代码单元中，你需要实现以下的功能：
 - 使用 `DataFrame.drop` 函数移除数据集中你选择的不需要的特征，并将移除后的结果赋值给 `new_data` 。
 - 使用 `sklearn.model_selection.train_test_split` 将数据集分割成训练集和测试集。
   - 使用移除的特征作为你的目标标签。设置 `test_size` 为 `0.25` 并设置一个 `random_state` 。
 
 
 - 导入一个 DecisionTreeRegressor （决策树回归器），设置一个 `random_state`，然后用训练集训练它。
 - 使用回归器的 `score` 函数输出模型在测试集上的预测得分。


```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
```


```python
features = data.columns.tolist()

for feature in features:
    # TODO：为DataFrame创建一个副本，用'drop'函数丢弃一个特征# TODO：
    new_data = data.drop(feature, axis=1)
    target = data[feature]

    # TODO：使用给定的特征作为目标，将数据分割成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(new_data, target, test_size=0.25, random_state=10)

    # TODO：创建一个DecisionTreeRegressor（决策树回归器）并在训练集上训练它
    regressor = DecisionTreeRegressor(random_state=10)
    regressor.fit(X_train, y_train)

    # TODO：输出在测试集上的预测得分
    score = regressor.score(X_test, y_test)
    print("%s在预测集上的预测得分：%.4f" % (feature, score))
```

    Fresh在预测集上的预测得分：-0.3792
    Milk在预测集上的预测得分：-0.4421
    Grocery在预测集上的预测得分：0.7238
    Frozen在预测集上的预测得分：0.0548
    Detergents_Paper在预测集上的预测得分：0.4944
    Delicatessen在预测集上的预测得分：-10.5627
    

### 问题 2

* 你尝试预测哪一个特征？
* 预测的得分是多少？
* 这个特征对于区分用户的消费习惯来说必要吗？为什么？  

**提示：** 决定系数（coefficient of determination），$R^2$ 结果在0到1之间，1表示完美拟合，一个负的 $R^2$ 表示模型不能够拟合数据。如果你对某个特征得到了低分，这使我们相信这一特征点是难以预测其余特征点的。当考虑相关性时，这使其成为一个重要特征。

**回答:**
预测特征及其得分：
* Fresh在预测集上的预测得分：-0.3792
* Milk在预测集上的预测得分：-0.4421
* Grocery在预测集上的预测得分：0.7238
* Frozen在预测集上的预测得分：0.0548
* Detergents_Paper在预测集上的预测得分：0.4944
* Delicatessen在预测集上的预测得分：-10.5627

特征Fresh、Milk和Delicatessen均在预测上取得了负的R2分数，说明将这个三个特征与其他特征具有较低的相关性，因此是比较重要的特征，对于区分用户的消费习惯很有必要；考虑到Grocery这一特征得分较高，说明该特征可以通过其他特征进行预测，这个特征也许不是重要的；另外两个特征Frozen和Detergents_Paper的R2分数虽然大于0，但是不高，因此不应该忽视。

**review:**

* 决定系数高 → 拟合度好 → 说明该变量可以通过其他变量来预测，那么该变量对于区分顾客消费习惯来说，是相对不必要的。
* 决定系数低 → 拟合度差 → 说明该变量不可以通过其他变量来预测，那么该变量对于区分顾客消费习惯来说，是相对必要的。

### 可视化特征分布
为了能够对这个数据集有一个更好的理解，我们可以对数据集中的每一个产品特征构建一个散布矩阵（scatter matrix）。如果你发现你在上面尝试预测的特征对于区分一个特定的用户来说是必须的，那么这个特征和其它的特征可能不会在下面的散射矩阵中显示任何关系。相反的，如果你认为这个特征对于识别一个特定的客户是没有作用的，那么通过散布矩阵可以看出在这个数据特征和其它特征中有关联性。运行下面的代码以创建一个散布矩阵。


```python
scatter = pd.plotting.scatter_matrix(data, alpha=0.5, figsize=(14, 8), diagonal='kde')
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512141241718.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)


### 问题 3

* 使用散布矩阵作为参考，讨论数据集的分布，特别是关于正态性，异常值，0附近的大量数据点等。如果你需要单独分离出一些图表来进一步强调你的观点，您也可以这样做。
* 是否存在具有某种程度相关性的特征对？
* 这个结果是验证了还是否认了你尝试预测的那个特征的相关性？
* 这些功能的数据如何分布？

**提示：** 数据是正态分布（normally distributed）吗？ 大多数数据点在哪里？ 您可以使用corr（）来获取特征相关性，然后使用热图（heatmap）将其可视化（输入到热图中的数据应该是有相关性的值，例如：data.corr（））以获得进一步的理解。


```python
import seaborn as sns
corr = data.corr()
display(corr)
ax = sns.heatmap(corr)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fresh</th>
      <td>1.000000</td>
      <td>0.100510</td>
      <td>-0.011854</td>
      <td>0.345881</td>
      <td>-0.101953</td>
      <td>0.244690</td>
    </tr>
    <tr>
      <th>Milk</th>
      <td>0.100510</td>
      <td>1.000000</td>
      <td>0.728335</td>
      <td>0.123994</td>
      <td>0.661816</td>
      <td>0.406368</td>
    </tr>
    <tr>
      <th>Grocery</th>
      <td>-0.011854</td>
      <td>0.728335</td>
      <td>1.000000</td>
      <td>-0.040193</td>
      <td>0.924641</td>
      <td>0.205497</td>
    </tr>
    <tr>
      <th>Frozen</th>
      <td>0.345881</td>
      <td>0.123994</td>
      <td>-0.040193</td>
      <td>1.000000</td>
      <td>-0.131525</td>
      <td>0.390947</td>
    </tr>
    <tr>
      <th>Detergents_Paper</th>
      <td>-0.101953</td>
      <td>0.661816</td>
      <td>0.924641</td>
      <td>-0.131525</td>
      <td>1.000000</td>
      <td>0.069291</td>
    </tr>
    <tr>
      <th>Delicatessen</th>
      <td>0.244690</td>
      <td>0.406368</td>
      <td>0.205497</td>
      <td>0.390947</td>
      <td>0.069291</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512141305417.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)


**回答:**
* 通过散步矩阵可以看出，单个特征的分布为**正偏态分布**，大部分数据均落在某一置信区间内（左下角），同时也可以看到有一定数量的异常点；
* 存在着一些比较明显的具有相关性的特征对，如Detergents_Paper和Grocery，Milk和Grocery, Milk和Detergents_Paper， 特征的相关系数和可视化热图可以进一步佐证这一关系，,它们的相关系数分别为0.9246，0.7283和0.6618， 这也证实了我上一步的预测，Grocery和Detergents_Paper这两个特征对于创建客户细分没有太大必要。

**review：**

![image.png](https://udacity-reviews-uploads.s3.amazonaws.com/_attachments/37879/1490886069/Screenshot_2016-05-15_at_11.06.51_AM.png)

## 数据预处理
在这个部分，你将通过在数据上做一个合适的缩放，并检测异常点（你可以选择性移除）将数据预处理成一个更好的代表客户的形式。预处理数据是保证你在分析中能够得到显著且有意义的结果的重要环节。

### 练习: 特征缩放
如果数据不是正态分布的，尤其是数据的平均数和中位数相差很大的时候（表示数据非常歪斜）。这时候通常用一个[非线性的缩放](https://github.com/czcbangkai/translations/blob/master/use_of_logarithms_in_economics/use_of_logarithms_in_economics.pdf)是很合适的，[（英文原文）](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) — 尤其是对于金融数据。一种实现这个缩放的方法是使用 [Box-Cox 变换](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html)，这个方法能够计算出能够最佳减小数据倾斜的指数变换方法。一个比较简单的并且在大多数情况下都适用的方法是使用自然对数。

在下面的代码单元中，你将需要实现以下功能：
 - 使用 `np.log` 函数在数据 `data` 上做一个对数缩放，然后将它的副本（不改变原始data的值）赋值给 `log_data`。 
 - 使用 `np.log` 函数在样本数据 `samples` 上做一个对数缩放，然后将它的副本赋值给 `log_samples`。


```python
# TODO：使用自然对数缩放数据
log_data = np.log(data)

# TODO：使用自然对数缩放样本数据
log_samples = np.log(samples)

# 为每一对新产生的特征制作一个散射矩阵
pd.plotting.scatter_matrix(log_data, alpha = 0.5, figsize = (14,8), diagonal = 'kde');
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/2019051214131946.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)


### 观察
在使用了一个自然对数的缩放之后，数据的各个特征会显得更加的正态分布。对于任意的你以前发现有相关关系的特征对，观察他们的相关关系是否还是存在的（并且尝试观察，他们的相关关系相比原来是变强了还是变弱了）。

运行下面的代码以观察样本数据在进行了自然对数转换之后如何改变了。


```python
display(log_data.corr())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fresh</th>
      <td>1.000000</td>
      <td>-0.019834</td>
      <td>-0.132713</td>
      <td>0.383996</td>
      <td>-0.155871</td>
      <td>0.255186</td>
    </tr>
    <tr>
      <th>Milk</th>
      <td>-0.019834</td>
      <td>1.000000</td>
      <td>0.758851</td>
      <td>-0.055316</td>
      <td>0.677942</td>
      <td>0.337833</td>
    </tr>
    <tr>
      <th>Grocery</th>
      <td>-0.132713</td>
      <td>0.758851</td>
      <td>1.000000</td>
      <td>-0.164524</td>
      <td>0.796398</td>
      <td>0.235728</td>
    </tr>
    <tr>
      <th>Frozen</th>
      <td>0.383996</td>
      <td>-0.055316</td>
      <td>-0.164524</td>
      <td>1.000000</td>
      <td>-0.211576</td>
      <td>0.254718</td>
    </tr>
    <tr>
      <th>Detergents_Paper</th>
      <td>-0.155871</td>
      <td>0.677942</td>
      <td>0.796398</td>
      <td>-0.211576</td>
      <td>1.000000</td>
      <td>0.166735</td>
    </tr>
    <tr>
      <th>Delicatessen</th>
      <td>0.255186</td>
      <td>0.337833</td>
      <td>0.235728</td>
      <td>0.254718</td>
      <td>0.166735</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


相关性变化：

features | before | after
:----: | :----: | :----:
Detergents_Paper和Grocery | 0.9246 | 0.7963
Milk和Grocery | 0.7283 | 0.7588
Milk和Detergents_Paper | 0.6618 | 0.6779


```python
# 展示经过对数变换后的样本数据
display(log_samples)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.702480</td>
      <td>10.901524</td>
      <td>10.925417</td>
      <td>8.959569</td>
      <td>10.092909</td>
      <td>8.774158</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.039983</td>
      <td>11.205013</td>
      <td>10.377047</td>
      <td>6.894670</td>
      <td>9.906981</td>
      <td>6.805723</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.627601</td>
      <td>10.296441</td>
      <td>9.806316</td>
      <td>9.725855</td>
      <td>8.506739</td>
      <td>9.053687</td>
    </tr>
  </tbody>
</table>
</div>


### 核密度估计
你可以使用 [seaborn库](http://seaborn.pydata.org/index.html) 中的代码，来实现对变化后每个特征 [核密度估计（KDE）](https://en.wikipedia.org/wiki/Kernel_density_estimation)的可视化。参见：[非参数估计：核密度估计KDE](https://blog.csdn.net/pipisorry/article/details/53635895)


```python
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_palette('Reds_r')
# sns.set_style('ticks')

# plot densities of log-transformed data
plt.figure(figsize=(8,4))
for col in data.columns:
    sns.kdeplot(log_data[col], shade=True)
plt.legend(loc=2)
```




    <matplotlib.legend.Legend at 0x1fc64ae9c50>



![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512141339920.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)


### 练习: 异常值检测
对于任何的分析，在数据预处理的过程中检测数据中的异常值都是非常重要的一步。异常值的出现会使得把这些值考虑进去后结果出现倾斜。这里有很多关于怎样定义什么是数据集中的异常值的经验法则。这里我们将使用[ Tukey 的定义异常值的方法](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/)：一个异常阶（outlier step）被定义成1.5倍的四分位距（interquartile range，IQR）。一个数据点如果某个特征包含在该特征的 IQR 之外的特征，那么该数据点被认定为异常点。

在下面的代码单元中，你需要完成下面的功能：
 - 将指定特征的 25th 分位点的值分配给 `Q1` 。使用 `np.percentile` 来完成这个功能。
 - 将指定特征的 75th 分位点的值分配给 `Q3` 。同样的，使用 `np.percentile` 来完成这个功能。
 - 将指定特征的异常阶的计算结果赋值给 `step`。
 - 选择性地通过将索引添加到 `outliers` 列表中，以移除异常值。

**注意：** 如果你选择移除异常值，请保证你选择的样本点不在这些移除的点当中！
一旦你完成了这些功能，数据集将存储在 `good_data` 中。

参考链接：[Python数据分析基础: 异常值检测和处理](https://segmentfault.com/a/1190000015926584)


```python
outlier_index = []
# 对于每一个特征，找到值异常高或者异常低的数据点
for feature in log_data.keys():
    
    # TODO: 计算给定特征的Q1（数据的25th分位点）
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: 计算给定特征的Q3（数据的75th分位点）
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: 使用四分位范围计算异常阶（1.5倍的四分位距）
    step = 1.5 * (Q3 - Q1)
    
    # plot the outlier
    print("Data points considered outliers for the feature ' {} ':".format(feature))
    display(log_data[~((log_data[feature] <= Q3 + step) & (log_data[feature] >= Q1 - step))])
    
    outlier_index += log_data[~((log_data[feature] <= Q3 + step) & (log_data[feature] >= Q1 - step))].index.tolist()
    
# TODO：列出有多个异常特征值的数据点
outlier = list(set([i for i in outlier_index if outlier_index.count(i) > 1]))
    
# TODO(可选): 选择你希望移除的数据点的索引
outliers = outlier

# 以下代码会移除outliers中索引的数据点, 并储存在good_data中
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)
```

    Data points considered outliers for the feature ' Fresh ':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
    </tr>
    <tr>
      <th>81</th>
      <td>5.389072</td>
      <td>9.163249</td>
      <td>9.575192</td>
      <td>5.645447</td>
      <td>8.964184</td>
      <td>5.049856</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1.098612</td>
      <td>7.979339</td>
      <td>8.740657</td>
      <td>6.086775</td>
      <td>5.407172</td>
      <td>6.563856</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3.135494</td>
      <td>7.869402</td>
      <td>9.001839</td>
      <td>4.976734</td>
      <td>8.262043</td>
      <td>5.379897</td>
    </tr>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>171</th>
      <td>5.298317</td>
      <td>10.160530</td>
      <td>9.894245</td>
      <td>6.478510</td>
      <td>9.079434</td>
      <td>8.740337</td>
    </tr>
    <tr>
      <th>193</th>
      <td>5.192957</td>
      <td>8.156223</td>
      <td>9.917982</td>
      <td>6.865891</td>
      <td>8.633731</td>
      <td>6.501290</td>
    </tr>
    <tr>
      <th>218</th>
      <td>2.890372</td>
      <td>8.923191</td>
      <td>9.629380</td>
      <td>7.158514</td>
      <td>8.475746</td>
      <td>8.759669</td>
    </tr>
    <tr>
      <th>304</th>
      <td>5.081404</td>
      <td>8.917311</td>
      <td>10.117510</td>
      <td>6.424869</td>
      <td>9.374413</td>
      <td>7.787382</td>
    </tr>
    <tr>
      <th>305</th>
      <td>5.493061</td>
      <td>9.468001</td>
      <td>9.088399</td>
      <td>6.683361</td>
      <td>8.271037</td>
      <td>5.351858</td>
    </tr>
    <tr>
      <th>338</th>
      <td>1.098612</td>
      <td>5.808142</td>
      <td>8.856661</td>
      <td>9.655090</td>
      <td>2.708050</td>
      <td>6.309918</td>
    </tr>
    <tr>
      <th>353</th>
      <td>4.762174</td>
      <td>8.742574</td>
      <td>9.961898</td>
      <td>5.429346</td>
      <td>9.069007</td>
      <td>7.013016</td>
    </tr>
    <tr>
      <th>355</th>
      <td>5.247024</td>
      <td>6.588926</td>
      <td>7.606885</td>
      <td>5.501258</td>
      <td>5.214936</td>
      <td>4.844187</td>
    </tr>
    <tr>
      <th>357</th>
      <td>3.610918</td>
      <td>7.150701</td>
      <td>10.011086</td>
      <td>4.919981</td>
      <td>8.816853</td>
      <td>4.700480</td>
    </tr>
    <tr>
      <th>412</th>
      <td>4.574711</td>
      <td>8.190077</td>
      <td>9.425452</td>
      <td>4.584967</td>
      <td>7.996317</td>
      <td>4.127134</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature ' Milk ':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86</th>
      <td>10.039983</td>
      <td>11.205013</td>
      <td>10.377047</td>
      <td>6.894670</td>
      <td>9.906981</td>
      <td>6.805723</td>
    </tr>
    <tr>
      <th>98</th>
      <td>6.220590</td>
      <td>4.718499</td>
      <td>6.656727</td>
      <td>6.796824</td>
      <td>4.025352</td>
      <td>4.882802</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
    <tr>
      <th>356</th>
      <td>10.029503</td>
      <td>4.897840</td>
      <td>5.384495</td>
      <td>8.057377</td>
      <td>2.197225</td>
      <td>6.306275</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature ' Grocery ':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature ' Frozen ':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38</th>
      <td>8.431853</td>
      <td>9.663261</td>
      <td>9.723703</td>
      <td>3.496508</td>
      <td>8.847360</td>
      <td>6.070738</td>
    </tr>
    <tr>
      <th>57</th>
      <td>8.597297</td>
      <td>9.203618</td>
      <td>9.257892</td>
      <td>3.637586</td>
      <td>8.932213</td>
      <td>7.156177</td>
    </tr>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
    </tr>
    <tr>
      <th>145</th>
      <td>10.000569</td>
      <td>9.034080</td>
      <td>10.457143</td>
      <td>3.737670</td>
      <td>9.440738</td>
      <td>8.396155</td>
    </tr>
    <tr>
      <th>175</th>
      <td>7.759187</td>
      <td>8.967632</td>
      <td>9.382106</td>
      <td>3.951244</td>
      <td>8.341887</td>
      <td>7.436617</td>
    </tr>
    <tr>
      <th>264</th>
      <td>6.978214</td>
      <td>9.177714</td>
      <td>9.645041</td>
      <td>4.110874</td>
      <td>8.696176</td>
      <td>7.142827</td>
    </tr>
    <tr>
      <th>325</th>
      <td>10.395650</td>
      <td>9.728181</td>
      <td>9.519735</td>
      <td>11.016479</td>
      <td>7.148346</td>
      <td>8.632128</td>
    </tr>
    <tr>
      <th>420</th>
      <td>8.402007</td>
      <td>8.569026</td>
      <td>9.490015</td>
      <td>3.218876</td>
      <td>8.827321</td>
      <td>7.239215</td>
    </tr>
    <tr>
      <th>429</th>
      <td>9.060331</td>
      <td>7.467371</td>
      <td>8.183118</td>
      <td>3.850148</td>
      <td>4.430817</td>
      <td>7.824446</td>
    </tr>
    <tr>
      <th>439</th>
      <td>7.932721</td>
      <td>7.437206</td>
      <td>7.828038</td>
      <td>4.174387</td>
      <td>6.167516</td>
      <td>3.951244</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature ' Detergents_Paper ':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
    </tr>
    <tr>
      <th>161</th>
      <td>9.428190</td>
      <td>6.291569</td>
      <td>5.645447</td>
      <td>6.995766</td>
      <td>1.098612</td>
      <td>7.711101</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature ' Delicatessen ':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
    </tr>
    <tr>
      <th>109</th>
      <td>7.248504</td>
      <td>9.724899</td>
      <td>10.274568</td>
      <td>6.511745</td>
      <td>6.728629</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>137</th>
      <td>8.034955</td>
      <td>8.997147</td>
      <td>9.021840</td>
      <td>6.493754</td>
      <td>6.580639</td>
      <td>3.583519</td>
    </tr>
    <tr>
      <th>142</th>
      <td>10.519646</td>
      <td>8.875147</td>
      <td>9.018332</td>
      <td>8.004700</td>
      <td>2.995732</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
    <tr>
      <th>183</th>
      <td>10.514529</td>
      <td>10.690808</td>
      <td>9.911952</td>
      <td>10.505999</td>
      <td>5.476464</td>
      <td>10.777768</td>
    </tr>
    <tr>
      <th>184</th>
      <td>5.789960</td>
      <td>6.822197</td>
      <td>8.457443</td>
      <td>4.304065</td>
      <td>5.811141</td>
      <td>2.397895</td>
    </tr>
    <tr>
      <th>187</th>
      <td>7.798933</td>
      <td>8.987447</td>
      <td>9.192075</td>
      <td>8.743372</td>
      <td>8.148735</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>203</th>
      <td>6.368187</td>
      <td>6.529419</td>
      <td>7.703459</td>
      <td>6.150603</td>
      <td>6.860664</td>
      <td>2.890372</td>
    </tr>
    <tr>
      <th>233</th>
      <td>6.871091</td>
      <td>8.513988</td>
      <td>8.106515</td>
      <td>6.842683</td>
      <td>6.013715</td>
      <td>1.945910</td>
    </tr>
    <tr>
      <th>285</th>
      <td>10.602965</td>
      <td>6.461468</td>
      <td>8.188689</td>
      <td>6.948897</td>
      <td>6.077642</td>
      <td>2.890372</td>
    </tr>
    <tr>
      <th>289</th>
      <td>10.663966</td>
      <td>5.655992</td>
      <td>6.154858</td>
      <td>7.235619</td>
      <td>3.465736</td>
      <td>3.091042</td>
    </tr>
    <tr>
      <th>343</th>
      <td>7.431892</td>
      <td>8.848509</td>
      <td>10.177932</td>
      <td>7.283448</td>
      <td>9.646593</td>
      <td>3.610918</td>
    </tr>
  </tbody>
</table>
</div>


### 问题 4

* 根据上述定义，是否有任何数据点有多个异常特征值？
* 是否应从数据集中删除这些数据点？
* 如果将任何数据点添加到要删除的异常值列表中，请解释原因。

**提示：** 如果你发现一些数据点，有多个类别中的异常值，请考虑可能的原因以及是否需要删除。 还要注意k-means如何受到异常值的影响，以及这是否会影响您是否删除异常值。



```python
display(log_data.iloc[outlier])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
    </tr>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
  </tbody>
</table>
</div>


**回答:**

* 由上可知，数据点128、65、66、75和154有多个异常特征值，这些异常点应该被删除；
* k-means和层次聚类对于异常值非常敏感，因为其要求将每一个点划分到一个簇中，优化目标是簇内差异最小化，因此即使单个噪音也会对整个簇产生较大扰动。

## 特征转换
在这个部分中你将使用主成分分析（PCA）来分析批发商客户数据的内在结构。由于使用PCA在一个数据集上会计算出最大化方差的维度，我们将找出哪一个特征组合能够最好的描绘客户。

### 练习: 主成分分析（PCA）

既然数据被缩放到一个更加正态分布的范围中并且我们也移除了需要移除的异常点，我们现在就能够在 `good_data` 上使用PCA算法以发现数据的哪一个维度能够最大化特征的方差。除了找到这些维度，PCA 也将报告每一个维度的解释方差比（explained variance ratio）--这个数据有多少方差能够用这个单独的维度来解释。注意 PCA 的一个组成部分（维度）能够被看做这个空间中的一个新的“特征”，但是它是原来数据中的特征构成的。

在下面的代码单元中，你将要实现下面的功能：
 - 导入 `sklearn.decomposition.PCA` 并且将 `good_data` 用 PCA 并且使用6个维度进行拟合后的结果保存到 `pca` 中。
 - 使用 `pca.transform` 将 `log_samples` 进行转换，并将结果存储到 `pca_samples` 中。


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=6, random_state=10)
pca.fit(good_data)

pca_samples = pca.transform(log_samples)

pca_result = vs.pca_results(good_data, pca)
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512141358822.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)



```python
def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''
    # Dimension indexing
    dimensions = ['Dimension {} '.format(i) for i in range(1, len(pca.components_) + 1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns=list(good_data.keys()))
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(ratios, columns=['Explained Variance'])
    variance_ratios.index = dimensions              

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize=(14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax=ax, kind='bar')
    ax.set_ylabel('Feature Weight')
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n    %.4f" % ev)

    pca_result = pd.concat([variance_ratios, components], axis=1)
    
    return pca_result
```


```python
display(pca_result.T)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dimension 1</th>
      <th>Dimension 2</th>
      <th>Dimension 3</th>
      <th>Dimension 4</th>
      <th>Dimension 5</th>
      <th>Dimension 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Explained Variance</th>
      <td>0.4430</td>
      <td>0.2638</td>
      <td>0.1231</td>
      <td>0.1012</td>
      <td>0.0485</td>
      <td>0.0204</td>
    </tr>
    <tr>
      <th>Fresh</th>
      <td>0.1675</td>
      <td>-0.6859</td>
      <td>-0.6774</td>
      <td>-0.2043</td>
      <td>-0.0026</td>
      <td>0.0292</td>
    </tr>
    <tr>
      <th>Milk</th>
      <td>-0.4014</td>
      <td>-0.1672</td>
      <td>0.0402</td>
      <td>0.0128</td>
      <td>0.7192</td>
      <td>-0.5402</td>
    </tr>
    <tr>
      <th>Grocery</th>
      <td>-0.4381</td>
      <td>-0.0707</td>
      <td>-0.0195</td>
      <td>0.0557</td>
      <td>0.3554</td>
      <td>0.8205</td>
    </tr>
    <tr>
      <th>Frozen</th>
      <td>0.1782</td>
      <td>-0.5005</td>
      <td>0.3150</td>
      <td>0.7854</td>
      <td>-0.0331</td>
      <td>0.0205</td>
    </tr>
    <tr>
      <th>Detergents_Paper</th>
      <td>-0.7514</td>
      <td>-0.0424</td>
      <td>-0.2117</td>
      <td>0.2096</td>
      <td>-0.5582</td>
      <td>-0.1824</td>
    </tr>
    <tr>
      <th>Delicatessen</th>
      <td>-0.1499</td>
      <td>-0.4941</td>
      <td>0.6286</td>
      <td>-0.5423</td>
      <td>-0.2092</td>
      <td>0.0197</td>
    </tr>
  </tbody>
</table>
</div>


## 主成分的解释
为了解释主成分，我们必须计算主成分与每一个原始特征的相关系数，因此会生成一个(n_components, n_features)的矩阵，这也就是PCA.components_。

由于标准化的原因，所有主成分的均值为0，还给出了每个成分的标准偏差，这些是特征值的平方根。

对主成分的解释是基于找出哪些变量与每个成分最强相关，即，这些数中的哪一个在**数量**上是大的，在任一方向上距离零最远。我们认为哪些数字大或小是当然是一个主观决定。您需要确定相关性在多大程度上的重要性。在此，**绝对值高于0.5的相关性被认为是重要的**。

### 问题 5

* 数据的第一个和第二个主成分**总共**解释了数据的多少变化？
* 前四个主成分呢？
* 使用上面提供的可视化图像，从用户花费的角度来讨论前四个主要成分中每个主成分代表的消费行为并给出你做出判断的理由。

**提示：**
* 对每个主成分中的特征分析权重的正负和大小。
* 结合每个主成分权重的正负讨论消费行为。
* 某一特定维度上的正向增长对应正权特征的增长和负权特征的减少。增长和减少的速率和每个特征的权重相关。[参考资料：Interpretation of the Principal Components](https://onlinecourses.science.psu.edu/stat505/node/54)

**回答:**

* 第一个和第二个主成分的可解释方差比率的和为0.7068， 也就是说这两个主成分解释了70.68%的数据变化；同理，前四个主成分解释了93.11%的数据变化；

* 对于第一个主成分来说，Detergents_Paper的特征贡献为-0.7514，同时，Milk和Grocery的贡献分别为-0.4014和-0.4381，这说明洗涤剂、牛奶和杂货的减小与第一个主成分的增加具有强相关，并且这三个特征呈正相关，这与咖啡馆或者杂货店的消费行为相似；

* 对于第二个主成分来说，按照相关程度依次是Fresh、Frozen和Delicateseen,并且这三个特征呈正相关，这与餐馆的消费行为类似；

* 对于第三个主成分来说，Delicateseen的增加和Fresh的减小具有相关性，一个买的多，一个买的少，并且这两个特征对主成分的贡献几乎相等，有点类似熟食店和生鲜店的消费行为；

* 对于第四个主成分来说，Frozen的增加和Delicateseen的减小具有相关性，Frozen对主成分的贡献较大，因而该主成分代表了冷冻店的消费行为；

* 由第一个和第二个主成分可见，六个特征根据彼此之间的关系可以被大致分成两组:(Detergents_Paper，Milk和Grocery)和(Fresh, Frozen和Delicateseen)

**review:**

所谓主成分分解，就是通过恰当的坐标变换，使得新坐标中的轴能够表达尽可能多的数据方差。最本质的，实际上某一 Dimension 就是原 Features 的**线性组合**，即

>D = w1 * F1 + w2 * F2 + ...

那么，你可以理解成，这代表了一种新的消费组合。如果某个 weight 绝对值很大，那么所对应的 feature 对该 Dimsension 影响很大。如果某两个 weight 同号，那么说明他们对这个 Dimension 的影响是相同的，反之则相反。

### 观察
运行下面的代码，查看经过对数转换的样本数据在进行一个6个维度的主成分分析（PCA）之后会如何改变。观察样本数据的前四个维度的数值。考虑这和你初始对样本点的解释是否一致。


```python
# 展示经过PCA转换的sample log-data
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_result.index.values))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dimension 1</th>
      <th>Dimension 2</th>
      <th>Dimension 3</th>
      <th>Dimension 4</th>
      <th>Dimension 5</th>
      <th>Dimension 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-4.3646</td>
      <td>-3.9519</td>
      <td>-0.1229</td>
      <td>0.6240</td>
      <td>0.5379</td>
      <td>0.0551</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-4.2903</td>
      <td>-1.4952</td>
      <td>-1.4997</td>
      <td>0.1394</td>
      <td>1.1469</td>
      <td>-0.6255</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.1899</td>
      <td>-4.8605</td>
      <td>0.0008</td>
      <td>0.4827</td>
      <td>0.5041</td>
      <td>-0.1988</td>
    </tr>
  </tbody>
</table>
</div>


### 练习：降维
当使用主成分分析的时候，一个主要的目的是减少数据的维度，这实际上降低了问题的复杂度。当然降维也是需要一定代价的：更少的维度能够表示的数据中的总方差更少。因为这个，**累计解释方差比（cumulative explained variance ratio）**对于我们确定这个问题需要多少维度非常重要。另外，如果大部分的方差都能够通过两个或者是三个维度进行表示的话，降维之后的数据能够被可视化。

在下面的代码单元中，你将实现下面的功能：
 - 将 `good_data` 用两个维度的PCA进行拟合，并将结果存储到 `pca` 中去。
 - 使用 `pca.transform` 将 `good_data` 进行转换，并将结果存储在 `reduced_data` 中。
 - 使用 `pca.transform` 将 `log_samples` 进行转换，并将结果存储在 `pca_samples` 中。


```python
# TODO：通过在good data上进行PCA，将其转换成两个维度
pca = PCA(n_components=2, random_state=10)
pca.fit(good_data)

# TODO：使用上面训练的PCA将good data进行转换
reduced_data = pca.transform(good_data)

# TODO：使用上面训练的PCA将log_samples进行转换
pca_samples = pca.transform(log_samples)

# 为降维后的数据创建一个DataFrame
reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
```

### 观察
运行以下代码观察当仅仅使用两个维度进行 PCA 转换后，这个对数样本数据将怎样变化。观察这里的结果与一个使用六个维度的 PCA 转换相比较时，前两维的数值是保持不变的。


```python
# 展示经过两个维度的PCA转换之后的样本log-data
display(pd.DataFrame(np.round(pca_samples, 4), columns=['Dimension 1', 'Dimension 2']))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dimension 1</th>
      <th>Dimension 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-4.3646</td>
      <td>-3.9519</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-4.2903</td>
      <td>-1.4952</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.1899</td>
      <td>-4.8605</td>
    </tr>
  </tbody>
</table>
</div>


## 联合核密度图
你可以考虑对变换后的数据进行可视化。使用seaborn库，你可以绘制分布的联合核密度图。


```python
g = sns.JointGrid("Dimension 1", "Dimension 2", reduced_data, xlim=(-6,6), ylim=(-5,5))
g = g.plot_joint(sns.kdeplot, cmp="Blues", shade=True)
g = g.plot_marginals(sns.kdeplot, shade=True)
```

    D:\Anaconda3\lib\site-packages\matplotlib\contour.py:1000: UserWarning: The following kwargs were not used by contour: 'cmp'
      s)
    
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512141417740.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)


## 可视化一个双标图（Biplot）
双标图是一个散点图，每个数据点的位置由它所在主成分的分数确定。坐标系是主成分（这里是 `Dimension 1` 和 `Dimension 2`）。此外，双标图还展示出初始特征在主成分上的投影。一个双标图可以帮助我们理解降维后的数据，发现主成分和初始特征之间的关系。

运行下面的代码来创建一个降维后数据的双标图。


```python
# 可视化双标图
vs.biplot(good_data, reduced_data, pca)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fc66d46438>



![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512141426685.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)



```python
def biplot(good_data, reduced_data, pca):
      
    fig, ax = plt.subplots(figsize = (14, 8))
    
    # scatterplot of the reduced data   
    ax.scatter(reduced_data.loc[:, 'Dimension 1'], reduced_data.loc[:, 'Dimension 2'], 
              facecolors='b', edgecolors='b', s=70, alpha=0.5)
    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16)

    feature_vectors = pca.components_.T
    
    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
                head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, good_data.columns[i], color = 'black', 
               ha='center', va='center', fontsize=18)
        
    return ax
```

### 观察

一旦我们有了原始特征的投影（红色箭头），就能更加容易的理解散点图每个数据点的相对位置。

在这个双标图中，哪些初始特征与第一个主成分有强关联？哪些初始特征与第二个主成分相关联？你观察到的是否与之前得到的 pca_results 图相符？

* Delicatessen、Fresh和Frozen与第一个主成分强关联；
* Detergent_Paper、Grocery和Milk与第二个主成分强关联；
* 这与我们之前观察到的结果一致。

 ## 聚类

在这个部分，你将选择使用 K-Means 聚类算法或者是高斯混合模型聚类算法以发现数据中隐藏的客户分类。然后，你将从簇中恢复一些特定的关键数据点，通过将它们转换回原始的维度和规模，从而理解他们的含义。

### 问题 6

* 使用 K-Means 聚类算法的优点是什么？
* 使用高斯混合模型聚类算法的优点是什么？
* 基于你现在对客户数据的观察结果，你选用了这两个算法中的哪一个，为什么？

    **提示：** 想一想硬聚类（hard clustering）和软聚类（soft clustering）间的区别，以及哪一种更适用于我们的数据

**回答:**
参考链接：[数据科学家必须了解的六大聚类算法：带你发现数据之美](https://www.jiqizhixin.com/articles/the-6-clustering-algorithms-data-scientists-need-to-know)
* K-Means聚类算法的优点：复杂度低，速度快，容易解释
* 高斯混合模型聚类算法的优点：GMMs 比 K-Means 在簇协方差方面更灵活；因为标准差参数，簇可以呈现任何椭圆形状，而不是被限制为圆形。K-Means 实际上是 GMM 的一个特殊情况，这种情况下每个簇的协方差在所有维度都接近 0。第二，因为 GMMs 使用概率，所以每个数据点可以有很多簇。因此如果一个数据点在两个重叠的簇的中间，我们可以简单地通过说它百分之 X 属于类 1，百分之 Y 属于类 2 来定义它的类。即 GMMs 支持混合资格。
* 基于先前观察，我们不能确切的将数据分到某一类中，因此选择高斯混合模型聚类算法。

### 练习: 创建聚类

针对不同情况，有些问题你需要的聚类数目可能是已知的。但是在聚类数目不作为一个**先验**知道的情况下，我们并不能够保证某个聚类的数目对这个数据是最优的，因为我们对于数据的结构（如果存在的话）是不清楚的。但是，我们可以通过计算每一个簇中点的**轮廓系数**来衡量聚类的质量。数据点的[轮廓系数](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)衡量了它与分配给他的簇的相似度，这个值范围在-1（不相似）到1（相似）。**平均**轮廓系数为我们提供了一种简单地度量聚类质量的方法。

在接下来的代码单元中，你将实现下列功能：
 - 在 `reduced_data` 上使用一个聚类算法，并将结果赋值到 `clusterer`，需要设置  `random_state` 使得结果可以复现。
 - 使用 `clusterer.predict` 预测 `reduced_data` 中的每一个点的簇，并将结果赋值到 `preds`。
 - 使用算法的某个属性值找到聚类中心，并将它们赋值到 `centers`。
 - 预测 `pca_samples` 中的每一个样本点的类别并将结果赋值到 `sample_preds`。
 - 导入 `sklearn.metrics.silhouette_score` 包并计算 `reduced_data` 相对于 `preds` 的轮廓系数。
   - 将轮廓系数赋值给 `score` 并输出结果。


```python
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# TODO：在降维后的数据上使用你选择的聚类算法
for n_components in [2, 3, 4, 5]:
    
    kmean = KMeans(n_clusters=n_components, random_state=10)
    gmm = GaussianMixture(n_components=n_components, random_state=10)
    clusters = (('kmean', kmean),('gmm', gmm))
    
    for name, clusterer in clusters:
        
        clusterer.fit(reduced_data)

        # TODO：预测每一个点的簇
        preds = clusterer.predict(reduced_data)

        # TODO：找到聚类中心
        if name == 'kmean':
            centers = clusterer.cluster_centers_
        else:
            centers = clusterer.means_
            
        # TODO：预测在每一个转换后的样本点的类
        sample_preds = clusterer.predict(pca_samples)

        # TODO：计算选择的类别的平均轮廓系数（mean silhouette coefficient）
        score = silhouette_score(reduced_data, preds)

        print("when the cluster is %s and the n_components is %d, the silhouette_score is %.4f." % (name, n_components, score))
```

    when the cluster is kmean and the n_components is 2, the silhouette_score is 0.4263.
    when the cluster is gmm and the n_components is 2, the silhouette_score is 0.4219.
    when the cluster is kmean and the n_components is 3, the silhouette_score is 0.3903.
    when the cluster is gmm and the n_components is 3, the silhouette_score is 0.3755.
    when the cluster is kmean and the n_components is 4, the silhouette_score is 0.3329.
    when the cluster is gmm and the n_components is 4, the silhouette_score is 0.2933.
    when the cluster is kmean and the n_components is 5, the silhouette_score is 0.3522.
    when the cluster is gmm and the n_components is 5, the silhouette_score is 0.3185.
    

### 问题 7

* 汇报你尝试的不同的聚类数对应的轮廓系数。
* 在这些当中哪一个聚类的数目能够得到最佳的轮廓系数？

**回答:**

cluster | n_clusters | silhouette_score
:----: | :----: | :----: 
Kmean | 2 | 0.4263
Kmean | 3 | 0.3903
Kmean | 4 | 0.3329
Kmean | 5 | 0.3522
GMM | 2 | 0.4219
GMM | 3 | 0.3755
GMM | 4 | 0.2933
GMM | 5 | 0.3285

当聚类数目为2时，Kmean和高斯混合模型聚类均取得最大的轮廓系数，分别为0.4263和0.4219

关于轮廓系数（silhouette_score），你可以参考这个[页面](https://blog.csdn.net/xueyingxue001/article/details/51966932)更细致地了解这个系数是怎么得到的，有什么意义。

### 聚类可视化
一旦你选好了通过上面的评价函数得到的算法的最佳聚类数目，你就能够通过使用下面的代码块可视化来得到的结果。作为实验，你可以试着调整你的聚类算法的聚类的数量来看一下不同的可视化结果。但是你提供的最终的可视化图像必须和你选择的最优聚类数目一致。


```python
n_components = 2
clusterer = GaussianMixture(n_components=n_components, covariance_type='full', random_state=10)
clusterer.fit(reduced_data)
preds = clusterer.predict(reduced_data)
centers = clusterer.means_

# 从已有的实现中展示聚类的结果
vs.cluster_results(reduced_data, preds, centers, pca_samples)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512141439761.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)


### 练习: 数据恢复
上面的可视化图像中提供的每一个聚类都有一个中心点。这些中心（或者叫平均点）并不是数据中真实存在的点，但是是所有预测在这个簇中的数据点的平均。对于创建客户分类的问题，一个簇的中心对应于那个分类的平均用户。因为这个数据现在进行了降维并缩放到一定的范围，我们可以通过施加一个反向的转换恢复这个点所代表的用户的花费。

在下面的代码单元中，你将实现下列的功能：
 - 使用 `pca.inverse_transform` 将 `centers` 反向转换，并将结果存储在 `log_centers` 中。
 - 使用 `np.log` 的反函数 `np.exp` 反向转换 `log_centers` 并将结果存储到 `true_centers` 中。


```python
# TODO：反向转换中心点
log_centers = pca.inverse_transform(centers)

# TODO：对中心点做指数转换
true_centers = np.exp(log_centers)

# 显示真实的中心点
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Segment 0</th>
      <td>8953.0</td>
      <td>2114.0</td>
      <td>2765.0</td>
      <td>2075.0</td>
      <td>353.0</td>
      <td>732.0</td>
    </tr>
    <tr>
      <th>Segment 1</th>
      <td>3552.0</td>
      <td>7837.0</td>
      <td>12219.0</td>
      <td>870.0</td>
      <td>4696.0</td>
      <td>962.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
import seaborn as sns
true_centers = true_centers.append(data.describe().loc['mean'])
true_centers.plot(kind='bar', figsize=(15,6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fc647342e8>



![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512141448716.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)


### 问题 8
* 考虑上面的代表性数据点在每一个产品类型的花费总数，你认为这些客户分类代表了哪类客户？为什么？需要参考在项目最开始得到的平均值来给出理由。

**提示：** 一个被分到`'Cluster X'`的客户最好被用 `'Segment X'`中的特征集来标识的企业类型表示。考虑每个细分所代表的选择特征点的值。 引用它们的各项平均值，以了解它们代表什么样的机构。

**回答:**

* Segment0的每一种产品花费数额均小于平均值，但是其fresh花费最多，其次是杂货、牛奶、冷冻产品等，因此Cluster 0应该属于餐馆咖啡馆等客户；
* Segment1的Milk、Grocery和Detergents_Paper都超过了平均值，Cluster 1应该是属于零售商；

### 问题 9
* 对于每一个样本点**问题 8 **中的哪一个分类能够最好的表示它？
* 你之前对样本的预测和现在的结果相符吗？

运行下面的代码单元以找到每一个样本点被预测到哪一个簇中去。


```python
# 显示预测结果
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)
```

    Sample point 0 predicted to be in Cluster 1
    Sample point 1 predicted to be in Cluster 1
    Sample point 2 predicted to be in Cluster 1
    

**回答:**

在之前的预测中：

* 第一个客户属于大型超市；
* 第二个客户属于咖啡饮料行业；
* 第三个客户属于餐馆行业

现在的预测：

* 第一个客户属于聚类1：零售商，符合之前的猜测
* 第二个客户属于聚类1：零售商，不符合之前的猜测
* 第三个客户属于聚类1：零售商， 不符合之前的猜测

## 结论

在最后一部分中，你要学习如何使用已经被分类的数据。首先，你要考虑不同组的客户**客户分类**，针对不同的派送策略受到的影响会有什么不同。其次，你要考虑到，每一个客户都被打上了标签（客户属于哪一个分类）可以给客户数据提供一个多一个特征。最后，你会把客户分类与一个数据中的隐藏变量做比较，看一下这个分类是否辨识了特定的关系。

### 问题 10
在对他们的服务或者是产品做细微的改变的时候，公司经常会使用 [A/B tests ](https://en.wikipedia.org/wiki/A/B_testing)以确定这些改变会对客户产生积极作用还是消极作用。这个批发商希望考虑将他的派送服务从每周5天变为每周3天，但是他只会对他客户当中对此有积极反馈的客户采用。

* 这个批发商应该如何利用客户分类来知道哪些客户对它的这个派送策略的改变有积极的反馈，如果有的话？你需要给出在这个情形下A/B 测试具体的实现方法，以及最终得出结论的依据是什么？

**提示：** 我们能假设这个改变对所有的客户影响都一致吗？我们怎样才能够确定它对于哪个类型的客户影响最大？

**回答：**
简单地了解了一下AB测试，当供应商要对现有的服务做出改进（每周5天发货变为每周3天发货）的时候，需要进行A/B测试，方法就是将客户分为2组，且这两组客户的数量相等，客户的特征相似（也就是每一种类型的客户在2组中的数量都差不多），因为我们之前已经通过GMM将客户分为2类，为了在影响尽可能少的客户的前提下得出结论，我们并不对所有客户都进行测试，我们挑选的测试的客户均来自于两个簇的中心，现在从这两个簇的中心各取相等数量的客户作为测试客户，然后从测试客户中每一类各取50%的客户放入第一组，剩余的客户放入第二组，这样分组就完成了。接下来分别对这两组的客户采取不同的发货策略，第一组采用之前的发货策略，第二组采用新的发货策略。如果在接下来的时间内，两组客户收到的投诉很接近甚至第二组的客户的投诉量少于第一组，那么说明发货周期的更改策略是可行的。如果第二组客户的投诉大于第一组，那么说明该发货策略对所有或者某一类客户是有影响的。接下来就分析第二组客户中投诉客户的类型，如果两种类型的客户的投诉量差不多，说明该策略的更改对所有客户都有影响。如果某种类型的客户的投诉量明显高于另一种类型，那么策略的更改对于这一类客户的影响更大。

### 问题 11
通过聚类技术，我们能够将原有的没有标记的数据集中的附加结构分析出来。因为每一个客户都有一个最佳的划分（取决于你选择使用的聚类算法），我们可以把用户分类作为数据的一个[工程特征](https://en.wikipedia.org/wiki/Feature_learning#Unsupervised_feature_learning)。假设批发商最近迎来十位新顾客，并且他已经为每位顾客每个产品类别年度采购额进行了预估。

* 进行了这些估算之后，批发商该如何运用它的预估和非监督学习的结果来对这十个新的客户进行更好的预测？

**提示**：在下面的代码单元中，我们提供了一个已经做好聚类的数据（聚类结果为数据中的cluster属性），我们将在这个数据集上做一个小实验。尝试运行下面的代码看看我们尝试预测‘Region’的时候，如果存在聚类特征'cluster'与不存在相比对最终的得分会有什么影响？这对你有什么启发？


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 读取包含聚类结果的数据
cluster_data = pd.read_csv('cluster.csv')
y = cluster_data['Region']
X = cluster_data.drop(['Region'], axis=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

clf = RandomForestClassifier(random_state=24)
clf.fit(X_train, y_train)
score_with_cluster = clf.score(X_test, y_test)

# 移除cluster特征
X_train = X_train.copy()
X_train.drop(['cluster'], axis=1, inplace=True)
X_test = X_test.copy()
X_test.drop(['cluster'], axis=1, inplace=True)

clf.fit(X_train, y_train)
score_no_cluster = clf.score(X_test, y_test)

print("不使用cluster特征的得分: %.4f"%score_no_cluster)
print("使用cluster特征的得分: %.4f"%score_with_cluster)
```

    不使用cluster特征的得分: 0.6437
    使用cluster特征的得分: 0.6667
    

    D:\Anaconda3\lib\site-packages\sklearn\ensemble\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    

**回答：**

结果表明，使用包含cluster特征的数据进行训练得到的模型取得了更好地分类效果。这就说明，预先使用聚类算法挖掘数据之间的潜在关系对于监督学习在于一定程度上是有帮助的，但是受限于数据本身的分布，并不一定总是有效。

**review:**

* 是的，聚类数据会使结果有所提升。
* 我们知道，监督学习和非监督学习解决的是两类工作。对于监督学习，它是通过(feature,label)的对来学习，最后预测label；对于非监督学习，它只通过feature本身来学习，最后能预测对应sample的label。
* 那么，在这个情景中，我们可以使用非监督学习的成果，即得到的label，来增强监督学习的结果：利用这个label加入监督学习的input feature，来给监督学习增维。

### 可视化内在的分布

在这个项目的开始，我们讨论了从数据集中移除 `'Channel'` 和 `'Region'` 特征，这样在分析过程中我们就会着重分析用户产品类别。通过重新引入 `Channel` 这个特征到数据集中，并施加和原来数据集同样的 PCA 变换的时候我们将能够发现数据集产生一个有趣的结构。

运行下面的代码单元以查看哪一个数据点在降维的空间中被标记为 `'HoReCa'` (旅馆/餐馆/咖啡厅)或者 `'Retail'`。另外，你将发现样本点在图中被圈了出来，用以显示他们的标签。


```python
# 根据‘Channel‘数据显示聚类的结果
vs.channel_results(reduced_data, outliers, pca_samples)
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512141459825.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)


### 问题 12

* 你选择的聚类算法和聚类点的数目，与内在的旅馆/餐馆/咖啡店和零售商的分布相比，有足够好吗？
* 根据这个分布有没有哪个簇能够刚好划分成'零售商'或者是'旅馆/饭店/咖啡馆'？
* 你觉得这个分类和前面你对于用户分类的定义是一致的吗？

**回答：**

根据内在的分布，数据集可以分成两个簇，这与我选择的聚类点的数目一致。GMM算法将大部分数据都正确区分，但是实际上可以看到，绿色的类别中包含了不少的红色样本，但是由于GMM预测的是某一点属于某一类的概率，其认为属于绿色的概率更大，因此将其划分成了绿色点，因此我们应该关注GMM对于每一个样本点赋予的概率。

从上图也可以看出，两个类别的中间部分其实很难作出区分，也就是说既可以判定成零售商也可以判定成旅馆/饭店/咖啡馆，大体上来说，与之前对于用户分明类的定义一致。
