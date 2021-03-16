
# 机器学习工程师纳米学位
## 模型评价与验证
## 项目 1: 预测波士顿房价


欢迎来到预测波士顿房价项目！在此文件中，我们已经提供了一些示例代码，但你还需要完善更多功能才能让项目成功运行。除非有明确要求，你无需修改任何已给出的代码。习题的标题为**编程练习**。每一部分都会有详细的指导，需要实现的部分也会在注释中以**TODO**标出。请仔细阅读所有的提示！

除了实现代码外，你还**必须**回答一些问题。请仔细阅读每个问题，并且在问题后的**'回答'**文字框中写出完整的答案。我们的项目审阅者将会根据你对问题的回答和撰写代码所实现的功能来进行评分。

>**提示：**代码 和 Markdown单元格 可通过 **Shift + Enter** 快捷键运行。另外，Markdown单元格  可以通过双击进入编辑模式。

---
## 第一步. 导入数据
在这个项目中，你将使用我们提供的波士顿房屋信息数据来训练和测试一个模型，并对模型的性能和预测能力进行评估。我们希望可以通过该模型实现对房屋的价值预估，提高房地产经纪人的工作效率。

此项目的数据集来自[kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)原始数据，未经过任何处理。该数据集统计了2006年至2010年波士顿个人住宅销售情况，包含2900多条观测数据（其中一半是训练数据，即我们的`housedata.csv`文件）。更多文档信息可以参考[作者的文档](http://jse.amstat.org/v19n3/decock.pdf)（可以不看），以及项目附件`data_description.txt`文件（特征描述文件，一定要看）。

运行下面区域的代码以载入一些此项目所需的Python库。如果成功返回提示语句，则说明载入成功。


```python
# 载入此项目需要的库
import numpy as np
import pandas as pd
import visuals as vs # Supplementary code 补充的可视化代码

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn') # use seaborn style 使用seaborn风格

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
print('你已经成功载入所有库！')
```

    你已经成功载入所有库！
    

### 编程练习 1：加载数据
你的第一个编程练习是加载波士顿房价数据。我们已为你导入了 ` Pandas `，你需要使用这个库中的[read_csv方法](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)来执行必要的过程。

导入数据将会是你整个项目的开始

如果成功返回数据集的大小，表示数据集已载入成功。


```python
# 1 TODO：载入波士顿房屋的数据集：使用pandas载入csv，并赋值到data_df
data_df = pd.read_csv('housedata.csv')

# 成功载入的话输出训练数据行列数目
print("Boston housing dataset has {} data points with {} variables each.".format(*data_df.shape))
```

    Boston housing dataset has 1460 data points with 81 variables each.
    

---
## 第二步. 数据分析
这个部分，你要对已有的波士顿房地产数据进行初步的观察与处理。同时，通过对数据的探索可以让你更好地理解数据。

由于这个项目的最终目标是建立一个预测房屋价值的模型，我们需要将数据集分为**特征(features)**和**目标变量(target variable)**。
- **目标变量**：` 'SalePrice'`，是我们希望预测的变量。
- **特征**：除` 'SalePrice'`外的属性都是特征，它们反应了数据点在某些方面的表现或性质。

### 编程练习 2：观察数据
你的第二个编程练习是对波士顿房价的数据进行观察,这一步会让你掌握更多数据本身的信息。具体问题如下：

**问题2.1：使用[head方法](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html)打印并观察前7条`data_df`数据**


```python
# 2.1 TODO: 打印出前7条data_df
print(data_df.head(7))
```

       Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
    0   1          60       RL         65.0     8450   Pave   NaN      Reg   
    1   2          20       RL         80.0     9600   Pave   NaN      Reg   
    2   3          60       RL         68.0    11250   Pave   NaN      IR1   
    3   4          70       RL         60.0     9550   Pave   NaN      IR1   
    4   5          60       RL         84.0    14260   Pave   NaN      IR1   
    5   6          50       RL         85.0    14115   Pave   NaN      IR1   
    6   7          20       RL         75.0    10084   Pave   NaN      Reg   
    
      LandContour Utilities  ... PoolArea PoolQC  Fence MiscFeature MiscVal  \
    0         Lvl    AllPub  ...        0    NaN    NaN         NaN       0   
    1         Lvl    AllPub  ...        0    NaN    NaN         NaN       0   
    2         Lvl    AllPub  ...        0    NaN    NaN         NaN       0   
    3         Lvl    AllPub  ...        0    NaN    NaN         NaN       0   
    4         Lvl    AllPub  ...        0    NaN    NaN         NaN       0   
    5         Lvl    AllPub  ...        0    NaN  MnPrv        Shed     700   
    6         Lvl    AllPub  ...        0    NaN    NaN         NaN       0   
    
      MoSold YrSold  SaleType  SaleCondition  SalePrice  
    0      2   2008        WD         Normal     208500  
    1      5   2007        WD         Normal     181500  
    2      9   2008        WD         Normal     223500  
    3      2   2006        WD        Abnorml     140000  
    4     12   2008        WD         Normal     250000  
    5     10   2009        WD         Normal     143000  
    6      8   2007        WD         Normal     307000  
    
    [7 rows x 81 columns]
    

**问题2.2：Id特征对我们训练数据没有任何用处，在`data_df`中使用[drop方法](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html)删除`'Id'`列数据**


```python
# 2.2 TODO: 删除data_df中的Id特征（保持数据仍在data_df中，不更改变量名）
data_df.drop(columns=['Id'], inplace=True)
```

**问题2.3：使用[describe方法](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)观察`data_df`各个特征的统计信息：**


```python
# 2.3 TODO:
data_df.describe(include=[np.number])
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
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>46.549315</td>
      <td>...</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>161.319273</td>
      <td>...</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>...</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 37 columns</p>
</div>




```python
# data_df.info() #查看dataFrame的信息
```

### 编程练习3： 数据预处理
我们的数据不可能是百分百的‘干净’数据（即有用数据），总会在采集整理时有些”失误“、“冗余”，造成“脏”数据，所以我们要从数据的正确性和完整性这两个方面来清理数据。

- **正确性**：一般是指有没有异常值，比如我们这个数据集中[作者的文档](http://jse.amstat.org/v19n3/decock.pdf)所说：
`I would recommend removing any houses with more than 4000 square feet from the data set (which eliminates these five unusual observations) before assigning it to students.`
建议我们去掉数据中`'GrLivArea'`中超过4000平方英尺的房屋（具体原因可以参考文档），当然本数据集还有其他的异常点，这里不再处理。
- **完整性**：采集或者整理数据时所产生的空数据造成了数据的完整性缺失，通常我们会使用一定的方法处理不完整的数据。在本例中，我们使用以下两种方法，一是[丢弃数据](https://discuss.analyticsvidhya.com/t/what-should-be-the-allowed-percentage-of-missing-values/2456),即选择丢弃过多空数据的特征（或者直接丢弃数据行，前提是NA数据占比不多），二是填补数据，填补的方法也很多，均值中位数众数填充等等都是好方法。

**问题3.1：正确性**

请按下述步骤删除一些不合理的数据

**问题3.1.1：使用`matplotlib`库中的[scatter方法](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html) 绘制`'GrLivArea'`和`'SalePrice'`的散点图，x轴为`'GrLivArea'`，y轴为`'SalePrice'`，观察数据**


```python
# 3.1.1 TODO:绘制要求的图形
plt.scatter(data_df['GrLivArea'], data_df['SalePrice'], c='blue', marker='.', s=30)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()
```


![png](output_18_0.png)


**问题3.1.2：通过上图我们可以看到那几个异常值，即`'GrLivArea'`大于4000，但是`'SalePrice'`又极低的数据，从`data_df`删除这几个异常值，删除后重新绘制`'GrLivArea'`和`'SalePrice'`的关系图，确认异常值已删除。**


```python
# 3.1.2 
# TODO:从data_df中删除 GrLivArea大于4000 且 SalePrice低于300000 的值
index = data_df[(data_df['GrLivArea'] > 4000) & (data_df['SalePrice'] < 300000)].index
data_df.drop(index=index, inplace=True)

# TODO:重新绘制GrLivArea和SalePrice的关系图，确认异常值已删除
plt.scatter(data_df['GrLivArea'], data_df['SalePrice'], c='blue', marker='.', s=50)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()

```


![png](output_20_0.png)


**问题3.2：完整性**

请按下述步骤，补足数据的完整性

**问题3.2.1：筛选出过多空数据的特征，我们这个项目定为筛选出有超过25%为空数据的特征**


```python
limit_percent = 0.25
limit_value = len(data_df) * limit_percent
# 3.2.1 TODO 统计并打印出超过25%的空数据的特征，你可以考虑使用isna()
list(data_df.columns[data_df.isna().sum() > limit_value])
```




    ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']



**如果你整理出的特征是`'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'`，那就说明你统计对了，接着我们查看`data_description.txt`文件，就会发现，这些并非一定是空缺数据，而没有游泳池，篱笆等也会用NA来表示，那么就不需要删除这些特征了，而是用`None`来填充`NA`数据。**



**问题3.2.2：根据`data_description.txt`特征描述,使用[fillna方法](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)填充空数据，具体哪一种数据需要填充什么已经整理好了，请按提示要求来进行填充**


```python
# 直接运行不用修改
# 确定所有空特征
missing_columns = list(data_df.columns[data_df.isnull().sum() != 0])
# 确定哪些是类别特征，哪些是数值特征
missing_numerical = list(data_df[missing_columns].dtypes[data_df[missing_columns].dtypes != 'object'].index)
missing_category = [i for i in missing_columns if i not in missing_numerical]
print("missing_numerical:",missing_numerical)
print("missing_category:",missing_category)
```

    missing_numerical: ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
    missing_category: ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
    


```python
data_df.mode() # 取众数，返回DataFrame
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
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20.0</td>
      <td>RL</td>
      <td>60.0</td>
      <td>7200.0</td>
      <td>Pave</td>
      <td>Grvl</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0.0</td>
      <td>Ex</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>2009.0</td>
      <td>WD</td>
      <td>Normal</td>
      <td>140000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>Fa</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>Gd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 80 columns</p>
</div>




```python
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
# 3.4.1 TODO：按需填补上面数据
# 我的实现：
# # 填充众数：
# mode = data_df[fill_Mode].mode().iloc[0,0]
# data_df.fillna(value={fill_Mode[0]: mode}, inplace=True)
# # 填充None：
# d_None={}
# for i in fill_None:
#     d_None[i] = 'None'
# data_df.fillna(value=d_None, inplace=True) 
# # 填充0：
# data_df[fill_0].fillna(value=0, inplace=True) 
# # 填充中位数：
# data_df[fill_median].fillna(data_df[fill_median].median(), inplace=True)

# 简化实现：
data_df[fill_Mode] = data_df[fill_Mode].fillna(data_df[fill_Mode].mode())
data_df[fill_None] = data_df[fill_None].fillna('None')
data_df[fill_0] = data_df[fill_0].fillna(0)
data_df[fill_median] = data_df[fill_median].fillna(data_df[fill_median].median())

# 验证
# data_df.isna().sum()
```

### 编程练习4： 特征分析
有这么一句话在业界广泛流传：特征数据决定了机器学习的上限，而模型和算法只是逼近这个上限而已。特征工程，是整个数据分析过程中不可缺少的一个环节，其结果质量直接关系到模型效果和最终结论。从上面两步中我们得到了“干净”的数据，但是`data_df`总共有81个特征，我们应当剔除那些无关紧要的特征（噪声），使用真正关键的特征来进行模型训练。现在需要我们对这些庞大的数据进行分析，提取出与目标最为关联的数据。





**问题4.1：绘制`'SalePrice'`的直方图，并说明该直方图属于什么[分布](https://zh.wikipedia.org/wiki/%E5%81%8F%E5%BA%A6)**


```python
# 4.1 TODO:绘制要求的图形
# plt.hist(data_df['SalePrice'], bins=50, normed=False, color=None)
# plt.xlabel('SalePrice')
# plt.show()
data_df['SalePrice'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22186a3e7b8>




![png](output_31_1.png)


回答问题4.1：`'SalePrice'`属于 **正偏态** 分布


```python
import scipy
scipy.stats.skew(data_df['SalePrice'])
```




    1.8793604459195012



**结果>0，代表正偏，从计算上也说明了'SalePrice'属于正偏态分布。**

如果特征极其多，很难清晰的看到特征与目标变量之间的关系，就需要利用统计知识来进行多变量分析了。我们常使用[热图heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)结合[corr方法](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html)来进行客观分析，热图Heatmap可以用颜色变化来反映变量之间的相关性二维矩阵或说相关性表格中的数据信息，它可以直观地将数据值的大小以定义的颜色深浅表示出来。这个项目，为了简化训练，我们以相关性绝对值大于0.5为界来选取我们需要的特征。


```python
# 不用修改直接运行
corrmat = data_df.corr().abs() #计算连续型特征之间的相关系数
#将于SalePrice的相关系数大于5的特征取出来，并按照SalePrice降序排列，然后取出对应的特征名，保存在列表中
top_corr = corrmat[corrmat["SalePrice"]>0.5].sort_values(by = ["SalePrice"], ascending = False).index 
cm = abs(np.corrcoef(data_df[top_corr].values.T)) #注意这里要转置，否则变成样本之间的相关系数，而我们要计算的是特征之间的相关系数
f, ax = plt.subplots(figsize=(20, 9))
sns.set(font_scale=1.3)
hm = sns.heatmap(cm, cbar=True, annot=True,
                 square=True, fmt='.2f', annot_kws={'size': 13}, 
                 yticklabels=top_corr.values, xticklabels=top_corr.values);
data_df = data_df[top_corr]
```


![png](output_36_0.png)


我们已经从庞大的特征群中筛选出了最相关的特征，也了解了我们目标数据的分布，那么接下来，我们从创造性方面来对我们的特征进行“改造”。
- **创造性**：创造性主要是说两种情况，一种是对现有数据的处理，比如对类别的独热编码（One-hotEncoder）或者标签编码（LabelEncoder），数值的区间缩放，归一化，标准化等等，另一种就是根据某一个或多个特征创造一个新的特征，例如某特征按组分类(groupby)后，或者某些特征组合后来创造新特征等等。

因为我们筛选出来的特征都为数值类型特征，所以我们只做标准化的操作：这个项目是一个回归类型的项目，而我们的回归算法对标准正态分步预测较为准确，从我们的目标数据可以看出数据是一个偏态分布，那么我们使用log将数据从偏态分布转换为标准正态分布，最后进行标准化。


```python
# 不要修改，直接运行
from scipy.special import boxcox1p
from sklearn.preprocessing import StandardScaler

data_df['SalePrice'] = np.log1p(data_df['SalePrice']) # np.log1p()函数将数据从偏态分布转换成标准正态分布
numeric_features = list(data_df.columns)
numeric_features.remove('SalePrice')
for feature in numeric_features:
    #all_data[feat] += 1
    data_df[feature] = boxcox1p(data_df[feature], 0.15)

scaler = StandardScaler()
scaler.fit(data_df[numeric_features])
data_df[numeric_features] = scaler.transform(data_df[numeric_features])
```

**笔记：**
1. **对目标变量的变换**np.loglp()：这个函数用来在数据预处理过程对偏度较大的数据进行平滑处理，使其更加符合高斯分布，有利于后续得到一个更好地结果，数学公式：ln(x+1).
2. **对数值类型的特征变量的变换**boxcox1p()：对于数值类型为数值的特征变量，也需要进行类似的变换。通过统计每一变量的skew值(偏度/偏态值，越接近0，越符合正态分布。大于0为正偏态，小于0为负偏态)，对绝对值大于0.5的进行boxcox变换。可以使线性回归模型满足线性性、独立性、方差齐性以及正态性的同时，又不丢失信息，此种变换称之为Box—Cox变换。$$y=\frac{x^{\lambda}-1}{\lambda},\lambda != 0 \\ y=log(x),\lambda == 0$$Box-Cox变换后，残差可以更好的满足正态性、独立性等假设前提，降低了伪回归的概率


---
## 第三步. 建立模型
在项目的第三步中，你需要了解必要的工具和技巧来让你的模型能够进行预测。用这些工具和技巧对每一个模型的表现做精确的衡量可以极大地增强你预测的可信度。

### 编程练习5：定义衡量标准
如果不能对模型的训练和测试的表现进行量化地评估，我们就很难衡量模型的好坏。通常我们会定义一些衡量标准，这些标准可以通过对某些误差或者拟合程度的计算来得到。在这个项目中，你将通过运算[决定系数](https://en.wikipedia.org/wiki/Coefficient_of_determination) $R^2$ 来量化模型的表现。模型的决定系数是回归分析中十分常用的统计信息，经常被当作衡量模型预测能力好坏的标准。

$R^2$ 的数值范围从0至1，表示**目标变量**的预测值和实际值之间的相关程度平方的百分比。一个模型的 $R^2$ 值为0还不如直接用**平均值**来预测效果好；而一个 $R^2$ 值为1的模型则可以对目标变量进行完美的预测。从0至1之间的数值，则表示该模型中目标变量中有百分之多少能够用**特征**来解释。模型也可能出现负值的 $R^2$，这种情况下模型所做预测有时会比直接计算目标变量的平均值差很多。



问题5.1:
在下方代码的 `performance_metric` 函数中，你要实现：
- 使用 `sklearn.metrics` 中的 [`r2_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) 来计算 `y_true` 和 `y_predict` 的 $R^2$ 值，作为对其表现的评判。
- 将他们的表现评分储存到 `score` 变量中。


```python
# 5.1 TODO: 引入 'r2_score'
from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: 计算 'y_true' 与 'y_predict' 的r2值
    score = r2_score(y_true, y_predict)
    
    # 返回这一分数
    return score
```

**笔记**：

R^2是评价模型表现的方法之一，每个机器学习模型的建立都要有相对应的评价指标，后面我们会学到更多的评价指标。不过R^2其实也有很多局限性需要注意 https://en.wikipedia.org/wiki/Coefficient_of_determination#Caveats

skearn对于常见的模型表现衡量方法也有详细的介绍。
http://scikit-learn.org/stable/modules/model_evaluation.html


问题 5.2:拟合程度

假设一个数据集有五个数据且某一模型做出下列目标变量的预测：

| 真实数值 | 预测数值 |
| :-------------: | :--------: |
| 3.0 | 2.5 |
| -0.5 | 0.0 |
| 2.0 | 2.1 |
| 7.0 | 7.8 |
| 4.2 | 5.3 |
*你觉得这个模型已成功地描述了目标变量的变化吗？如果成功，请解释为什么，如果没有，也请给出原因。*  

**提示1**：运行下方的代码，使用 `performance_metric` 函数来计算 `y_true` 和 `y_predict` 的决定系数。

**提示2**：$R^2$ 分数是指可以从自变量中预测的因变量的方差比例。 换一种说法：

* $R^2$ 为0意味着因变量不能从自变量预测。
* $R^2$ 为1意味着可以从自变量预测因变量。
* $R^2$ 在0到1之间表示因变量可预测的程度。
* $R^2$ 为0.40意味着 Y 中40％的方差可以从 X 预测。


```python
#TODO 5.2：计算这一模型的表现
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))
```

    Model has a coefficient of determination, R^2, of 0.923.
    

问题 5.2 - 回答:**模型可以描述目标变量的变化，因为R^2值为0.923，说明自变量（预测数值）对因变量（真实数值）的解释越好。**

### 编程练习 6: 数据分割与重排
接下来，你需要分割波士顿房屋数据集，包括特征与目标变量、训练集和测试集。通常在这个过程中，数据也会被重排，以消除数据集中由于顺序而产生的偏差。
在下面的代码中，你需要



**问题6.1：将`data_df`分割为特征和目标变量**


```python
# TODO: 6.1 

labels = data_df['SalePrice']#TODO：提取SalePrice作为labels
features = data_df.drop(['SalePrice'], axis=1)#TODO：提取除了SalePrice以外的特征赋值为features
```

问题6.2 ：
* 使用 `sklearn.model_selection` 中的 [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)， 将 `features` 和 `prices` 的数据都分成用于训练的数据子集和用于测试的数据子集。
  - 分割比例为：80%的数据用于训练，20%用于测试；
  - 选定一个数值以设定 `train_test_split` 中的 `random_state` ，这会确保结果的一致性；
* 将分割后的训练集与测试集分配给 `X_train`, `X_test`, `y_train` 和 `y_test`。


```python
# TODO: 引入 'train_test_split'
from sklearn.model_selection import train_test_split

# TODO: 打乱并分割训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

# 成功~
print("Training and testing split was successful.")
```

    Training and testing split was successful.
    

问题 6.3 - 训练及测试
*将数据集按一定比例分为训练用的数据集和测试用的数据集对学习算法有什么好处？*

*如果用模型已经见过的数据，例如部分训练集数据进行测试，又有什么坏处？*

**提示：** 如果没有数据来对模型进行测试，会出现什么问题？

问题 6.3 - 回答:
- 将数据集按一定比例分为训练用的数据集和测试用的数据集对学习算法可以对算法进行验证，在一定程度上**避免算法过拟合**;

- 如果部分训练数据进行测试，会使得算法的**准确率不可靠**；

- 如果没有数据进行测试，**无法说明模型的优劣**。

- 将数据集分成训练集和测试集的**好处**：既可以用于训练又可以用于测试，而且不会相互干扰，而且可以对训练模型进行有效的验证。

- 用部分训练集进行测试的**坏处**：模型就是根据训练集得出的，使用训练集进行测试肯定会得出较好的结果，这不能判断训练模型的优劣。

---
## 第四步. 分析模型的表现
在项目的第四步，我们来观察不同参数下，模型在训练集和验证集上的表现。这里，我们专注于一个特定的算法（带剪枝的决策树[`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)，但这并不是这个项目的重点）和这个算法的一个参数 `'max_depth'`。用全部训练集训练，选择不同`'max_depth'` 参数，观察这一参数的变化如何影响模型的表现。画出模型的表现来对于分析过程十分有益。

### 学习曲线
下方区域内的代码会输出四幅图像，它们是一个决策树模型在不同最大深度下的表现。每一条曲线都直观得显示了随着训练数据量的增加，模型学习曲线在训练集评分和验证集评分的变化，评分使用决定系数 $R^2$。曲线的阴影区域代表的是该曲线的不确定性（用标准差衡量）。

运行下方区域中的代码，并利用输出的图形回答下面的问题。


```python
# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, labels)
```


![png](output_58_0.png)


### 问题 7 - 学习曲线
* 选择上述图像中的其中一个，并给出其最大深度。
* 随着训练数据量的增加，训练集曲线的评分有怎样的变化？验证集曲线呢？
* 如果有更多的训练数据，是否能有效提升模型的表现呢？

**提示：**学习曲线的评分是否最终会收敛到特定的值？一般来说，你拥有的数据越多，模型表现力越好。但是，如果你的训练和测试曲线以高于基准阈值的分数收敛，这是否有必要？基于训练和测试曲线已经收敛的前提下，思考添加更多训练点的优缺点。

问题 7 - 回答:**当max_depth为3时，随着训练数据的增加，训练集曲线的评分降低，验证集曲线的评分提高，但是两者会收敛到一特定数值，随着训练数据的增加，模型表现不再提高，此时没有必要再继续增加训练数据，因为会增加模型的训练时间。但是，如果在过拟合的情况下，比方说当max_depth为6或者10时，随着训练数据的增加，模型表现也不断提高，此时反而有必要增加训练数据。**

**笔记**：
对训练曲线和测试曲线趋势和意义解释的很好。这里随着数据的增多，max_depth不变的情况下，模型提升的幅度也越来越小。
![image.png](attachment:image.png)

传统的机器学习算法（又被称为基于统计的机器学习）在数据量达到一定程度后，更多的数据无法提升模型的表现。深度学习的一个优势就是它可以把大量的数据利用起来，提升学习表现。

这里还有更多关于学习曲线的介绍：
https://www.coursera.org/learn/machine-learning/lecture/Kont7/learning-curves
http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
传统的机器学习算法（又被称为基于统计的机器学习）在数据量达到一定程度后，更多的数据无法提升模型的表现。深度学习的一个优势就是它可以把大量的数据利用起来，提升学习表现。

这里还有更多关于学习曲线的介绍：
https://www.coursera.org/learn/machine-learning/lecture/Kont7/learning-curves
http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


### 复杂度曲线
下列代码内的区域会输出一幅图像，它展示了一个已经经过训练和验证的决策树模型在不同最大深度条件下的表现。这个图形将包含两条曲线，一个是训练集的变化，一个是验证集的变化。跟**学习曲线**相似，阴影区域代表该曲线的不确定性，模型训练和测试部分的评分都用的 `performance_metric` 函数。

**运行下方区域中的代码，根据问题5与问题6对输出的图形进行分析并给出你思考的结果。**


```python
vs.ModelComplexity(X_train, y_train)
```


![png](output_63_0.png)


### 问题 8 - 偏差（bias）与方差（variance）之间的权衡取舍
* 当模型以最大深度 1训练时，模型的预测是出现很大的偏差还是出现了很大的方差？
* 当模型以最大深度10训练时，情形又如何呢？
* 图形中的哪些特征能够支持你的结论？
  
**提示：** 高偏差表示欠拟合（模型过于简单），而高方差表示过拟合（模型过于复杂，以至于无法泛化）。考虑哪种模型（深度1或10）对应着上述的情况，并权衡偏差与方差。

问题 8 - 回答:**当max_depth为1时，模型表现出高偏差，因为训练集和测试集的R2分数都比较低；当max_depth为10时，模型表现出高方差，因为训练集的R2分数较高，而验证集的R2分数较低。**

**笔记**：

偏差-方差分解是解释学习算法泛化性能的重要工具。这里你需要理解三个方面：

**什么是泛化误差(Generalization error)：**

* 如果学得到的模型是 f ，那么用这个模型对测试数据预测的误差即为泛化误差。一般所说的泛化误差都是算法的**期望泛化误差**。

**Bias-Variance分解:**

* 现在我们来看题目中的方差和偏差到底是什么意思？通过简单的多项式展开合并，可以对算法的期望泛化误差进行分解可以得到(具体推到过程请参考[1]、[2]) :
![image.png](attachment:image.png)

* 也就是说：泛化误差=偏差+方差+噪声

* **偏差**度量了学习算法的期望预测和真实结果的偏离程度，即刻画了**学习算法本身的拟合能力**。

* **方差**度量了同样大小的训练集的变动所导致的学习性能的变化, 即 刻画了**数据扰动所造成的影响**。

**Bias-Variance trade-off:**

一般情况下，偏差和方差是有冲突的，如下图所示。为了取得好的泛化性能，则需要偏差较小，即能够充分拟合数据，并且方差较小，即使得数据扰动产生的影响小。

* **欠拟合**：在训练不足时，学习的拟合能力不够强，训练数据的扰动不足以使学习器产生显著变化，偏差主导了泛化错误率。
* **过拟合**：充分训练后, 学习器的拟合能力已非常强, 训练数据的轻微扰动都会导致学习器发生显著变化, 训练数据自身的、非全局的特性被学习器学到了，方差主导了泛化错误率。
![image.png](attachment:image.png)

更多内容请阅读参考资料。

参考资料：

[1] «机器学习», 周志华, 2.5 节偏差与方差.

[2] Understanding the Bias-Variance Tradeoff


### 问题 9- 最优模型的猜测
* 结合复杂度曲线，你认为最大深度是多少的模型能够最好地对未见过的数据进行预测？
* 你得出这个答案的依据是什么？

**提示**：查看问题8上方的图表，并查看模型在不同 `depth`下的验证分数。随着深度的增加模型的表现力会变得更好吗？我们在什么情况下获得最佳验证分数而不会使我们的模型过度复杂？请记住，奥卡姆剃刀：“在竞争性假设中，应该选择假设最少的那一个。”

问题 9 - 回答:

- **随着深度的增加模型表现并没有越来越好，验证集在max_depth为4时，模型收敛，并且训练R2分数和验证R2较高**。
- **随着最大深度的提升，训练集R2分数不断上升，验证集R2分数并没有得到提高，反而有下降的趋势，说明随着模型变得复杂，模型过拟合。**
- **考虑模型的复杂度和R2分数，最大深度为4的模型具有最好的泛化能力。**

---
## 第五步. 评估模型的表现
在项目的最后一节中，你将构建一个模型，并使用 `fit_model` 中的优化模型去预测客户特征集。

### 问题 10- 网格搜索（Grid Search）
* 什么是网格搜索法？
* 如何用它来优化模型？

**提示**：在解释网格搜索算法时，首先要理解我们为什么使用网格搜索算法，以及我们使用它的最终目的是什么。为了使你的回答更具有说服力，你还可以给出一个模型中可以使用此方法进行优化参数的示例。

问题 10 - 回答: **网格搜索法是给定参数值，穷举搜索寻找最优参数的算法。网格搜索法通过构建一个参数的字典，创造出不同的参数组合，输入给分类器进行训练，采用交叉验证，寻找验证分数最高的那一组参数，这一组参数就是模型的最优参数。**
- 网格搜索算法是一种穷举搜索的算法。
- 在所有候选的参数选择中，通过循环遍历，尝试每一种可能性，表现最好的参数就是最终的结果。其原理就像是在数组里找最大值。以有两个参数的模型为例，参数a有3种可能，参数b有4种可能，把所有可能性列出来，可以表示成一个3*4的表格，其中每个cell就是一个网格，循环过程就像是在每个网格里遍历、搜索，所以叫grid search。

### 问题 11 - 交叉验证
- 什么是K折交叉验证法（k-fold cross-validation）？
- [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 是如何结合交叉验证来完成对最佳参数组合的选择的？
- [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 中的`'cv_results_'`属性能告诉我们什么？
- 网格搜索为什么要使用K折交叉验证？K折交叉验证能够避免什么问题？

**提示**：在解释k-fold交叉验证时，一定要理解'k'是什么，和数据集是如何分成不同的部分来进行训练和测试的，以及基于'k'值运行的次数。
在考虑k-fold交叉验证如何帮助网格搜索时，你可以使用特定的数据子集来进行训练与测试有什么缺点，以及K折交叉验证是如何帮助缓解这个问题。

问题 11 - 回答：
- K折交叉验证是随机将训练集划分成K份，依次将其中的一份作为验证集，其余的K-1份作为训练集，得到K个模型，观察每个模型的表现，选择模型表现最好的那一个，K折交叉验证能够有效的防止过拟合；
- GridSearchCV通过交叉验证得到每个参数组合的得分，以此确定最优的参数组合；
- GridSearchCV 中的cv_results 属性返回一个字典，记录了每一组网格参数每一次训练/验证( K 折对应 K 次)对应的训练结果，包括训练/验证时间、训练/验证评估分数以及相关时间和评分的统计信息；
- 采用交叉验证可以使得网格搜索的结果可信度更高，对网格搜索来说，可以将训练数据可以按一定比例分为训练集和验证集，而不使用交叉验证，但是这样做的话，网格搜索的结果依赖于训练集和验证集的划分，这样的划分不具有代表性。采用K折交叉验证取平均值可以避免样本划分不合理的情况，使得结果可信度更高。

### 编程练习 7：训练最优模型
在这个练习中，你将需要将所学到的内容整合，使用**决策树算法**训练一个模型。为了得出的是一个最优模型，你需要使用网格搜索法训练模型，以找到最佳的 `'max_depth'` 参数。你可以把`'max_depth'` 参数理解为决策树算法在做出预测前，允许其对数据提出问题的数量。决策树是**监督学习算法**中的一种。

在下方 `fit_model` 函数中，你需要做的是：
1. **定义 `'cross_validator'` 变量**: 使用 `sklearn.model_selection` 中的 [`KFold`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) 创建一个交叉验证生成器对象;
2. **定义 `'regressor'` 变量**: 使用  `sklearn.tree` 中的 [`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) 创建一个决策树的回归函数;
3. **定义 `'params'` 变量**: 为 `'max_depth'` 参数创造一个字典，它的值是从1至10的数组;
4. **定义 `'scoring_fnc'` 变量**: 使用 `sklearn.metrics` 中的 [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)  创建一个评分函数；
 将 `‘performance_metric’` 作为参数传至这个函数中；
5. **定义 `'grid'` 变量**: 使用 `sklearn.model_selection` 中的 [`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 创建一个网格搜索对象；将变量`'regressor'`, `'params'`, `'scoring_fnc'`和 `'cross_validator'` 作为参数传至这个对象构造函数中；
  
如果你对python函数的默认参数定义和传递不熟悉，可以参考这个MIT课程的[视频](http://cn-static.udacity.com/mlnd/videos/MIT600XXT114-V004200_DTH.mp4)。


```python
# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    cross_validator = KFold(n_splits=10)
    
    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor(random_state=1)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':[i for i in range(1, 11)]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cross_validator' respectively.
    grid = GridSearchCV(regressor, params, scoring_fnc, cv = cross_validator)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_
```

## 第六步. 做出预测
当我们用数据训练出一个模型，它就可用于对新的数据进行预测。在我们的例子--决策树回归函数中，模型已经学会对新输入的数据“提问”，并返回对**目标变量**的预测值。你可以用这些预测来获取未知目标变量的数据的信息，但是，输入的新数据必须不能是已有训练数据之中的。

### 问题 12 - 最优模型
*最优模型的最大深度（maximum depth）是多少？此答案与你在**问题 9**所做的猜测是否相同？*

运行下方区域内的代码，将决策树回归函数代入训练数据的集合，以得到最优化的模型。


```python
# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
```

    Parameter 'max_depth' is 5 for the optimal model.
    

最终，使用我们确认好的参数来对测试数据进行预测，完成下面的问题，来看看我们的训练结果如何吧

**问题12.1：填入上题所确认的最优参数，查看测试结果**


```python
depth = 5
regressor = DecisionTreeRegressor(max_depth = depth)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test) 
score = performance_metric(y_test, y_pred)
print("The R2 score is ",score)
```

    The R2 score is  0.7520883029841192
    

**问题12.2：你刚刚计算了最优模型在测试集上的决定系数，你会如何评价这个结果？还有什么地方可以改进，以提升这一分数呢？**

回答问题12.2：这个结果并不是理想的，应该还需要利用决策树的其他参数进行网格搜索，以及使用更多的特征；
