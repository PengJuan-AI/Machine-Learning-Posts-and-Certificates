# 【机器学习应用】【Python】机器学习好帮手——sklearn

## 什么是scikit-learn (sklearn)

Scikit-learn是一个开源的机器学习库，它为机器学习中的模型拟合、评估，数据预处理等模块提供各种各样的工具。本文将介绍sklearn在机器学习中的一些简单应用。

## 安装sklearn
`pip install -U scikit-learn`

## 使用sklearn进行数据预处理
sklearn常见的数据预处理有Standardize，Normalize，MinMaxScaler，Ordinal Encoding和OneHot Encoding。

```python
from sklearn import preprocessing
import numpy as np

# 创建scaler
standard_scarler = preprocessing.StandardScaler()
normalize_scaler = preprocessing.Normalizer()
min_max_scaler = preprocessing.MinMaxScaler()

data = np.array([[1,2,3],[2,3,4],[3,4,5]])
print(f"Original data:\n{data}")

standarded_data = standard_scarler.fit_transform(data)
normalized_data = normalize_scaler.fit_transform(data)
min_max_scaled_data = min_max_scaler.fit_transform(data)

print(f"Standardized data:\n {standarded_data}")
print(f"Normalized data:\n {normalized_data}")
print(f"Min_max scaled data:\n {min_max_scaled_data}")
```
输出:
```
Original data:
[[1 2 3]
 [2 3 4]
 [3 4 5]]
Standardized data:
 [[-1.22474487 -1.22474487 -1.22474487]
 [ 0.          0.          0.        ]
 [ 1.22474487  1.22474487  1.22474487]]
Normalized data:
 [[0.26726124 0.53452248 0.80178373]
 [0.37139068 0.55708601 0.74278135]
 [0.42426407 0.56568542 0.70710678]]
Min_max scaled data:
 [[0.  0.  0. ]
 [0.5 0.5 0.5]
 [1.  1.  1. ]]
```
将类型数据转化成numeric data，我们可以使用`OrdinalEncoding`或`OneHotEncoding`。

`OrdinalEncoding`是根据类型的多少给每一类数据按顺序标号。
```python
ordinal_enc = preprocessing.OrdinalEncoder()

# 数据类别分别为颜色和大小，颜色有3种，大小有两种
# encoder从遇见的第一类数据开始标号，从0号开始
data = np.array([['green', 'big'],['red', 'big'],['red', 'small'],['yellow', 'small']])
encoded_data = ordinal_enc.fit_transform(data)

print(f"Original data:\n {data}")
print(f"Ordinal
```
输出：
```
Original data:
 [['green' 'big']
 ['red' 'big']
 ['red' 'small']
 ['yellow' 'small']]
Ordinal encoded data:
 [[0. 0.]
 [1. 0.]
 [1. 1.]
 [2. 1.]]
```
`OneHotEncoding`主要是将类型数据转换成由0和1组成的表。
```python
one_hot_enc  = preprocessing.OneHotEncoder()

# 数据类别分别为颜色和大小，颜色有3种，大小有两种
# OneHot encoder会将类别编码为nxn的表（n代表类别数目）
data = np.array([['green', 'big'],['red', 'big'],['red', 'small'],['yellow', 'small']])
encoded_data = one_hot_enc.fit_transform(data).toarray()

print(f"Original data:\n {data}")
print(f"One-hot encoded data:\n {np.array(encoded_data)}")
```
输出：
```
Original data:
 [['green' 'big']
 ['red' 'big']
 ['red' 'small']
 ['yellow' 'small']]
One-hot encoded data:
 [[1. 0. 0. 1. 0.]
 [0. 1. 0. 1. 0.]
 [0. 1. 0. 0. 1.]
 [0. 0. 1. 0. 1.]]
```

> ⚠️需要注意的是，使用scaler的时候要在train data上拟合(fit)，然后再`scaler.transform(test_data)`。在test data上`scaler.fi()`会导致test data的数据“泄漏”，从而影响模型的效果。

更多`sklearn.preprocessing`数据预处理的的模块参考[preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
### train_test_split
通常情况下，我们拿到的数据需要我们手动去分成训练集和测试集，sklearn中的train_test_split可帮助将原数据集分成训练集和测试集。

> 当我们手中没有可用数据时，可用`sklearn.datasets`获取示例数据集。`sklearn.datasets`详细用法可参见：[机器学习 - sklearn 自带数据集详细解释](https://zhuanlan.zhihu.com/p/618818240?utm_id=0)

```python
# train_test_split
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 以iris数据集为例
# 还可以通过在train_test_split中设置测试集的大小，默认是总数据集的25%
data = load_iris()
print(f"Size of iris data: {len(data.data)}")
print(f"Targets names: {data.target_names}")
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print(f"Size of train data: {len(X_train)}")
print(f"Size of test data: {len(X_test)}")
```
```
Size of iris data: 150
Targets names: ['setosa' 'versicolor' 'virginica']
Size of train data: 112
Size of test data: 38
```
`train_test_split`会随机切分训练集和测试集，因此设定`random_state`能够确保我们拟合时使用的都是同样的数据集，保证模型的稳定性。

## 使用sklearn导入模型
sklearn之所以是机器学习应用的好帮手，是因为它给我们提供了几乎所有的机器学习算法，我们也就不用从0开始写算法，可直接调用sklearn给我们准备好的算法模块。

以K近邻(KNN)算法为例 (结合前文处理的iris数据)
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# 可以设定近邻数(n_neighbors)，默认是5
model = KNeighborsClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 检查一下预测的结果
target_names = data.target_names
for y1, y2 in zip(y_test, y_pred):
    print(f"Actual names: {target_names[y1]} (Predicted: {target_names[y2]})")
```
输出：
```
Actual names: versicolor (Predicted: versicolor)
Actual names: setosa (Predicted: setosa)
Actual names: virginica (Predicted: virginica)
Actual names: versicolor (Predicted: versicolor)
Actual names: versicolor (Predicted: versicolor)
Actual names: setosa (Predicted: setosa)
...
```

KNN更详细的内容参考：[Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)


## 使用sklearn评估模型

模型训练完成之后，我们还要评估模型的表现，根据模型的结果再回到处理数据或调整超参数的步骤，进一步提高模型的表现。

我们可以直接用`model.score(X_test, y_test)`来评估模型的准确度。

也可以用`sklearn.metrics`给我们提供的工具。对于分类问题，有accuracy，precision rate和recall rate等指标。对于回归问题，r2_score，MAE(mean absolute error)等指标。详细可参考[`sklearn.metrics`官方文档](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)

### accuracy_score
结合前面训练的模型`model`，我们可以计算模型的准确度
```python
from sklearn.metrics import accuracy_score

# accuracy_score(y_true, y_pred)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")
print(f"Accuracy: {model.score(X_test, y_test)}")
```
输出：
```
Accuracy: 1.0
Accuracy: 1.0
```
哇，100%准确率～👏👏

当然，越高准确度某种程度上说明模型效果越好，但大多数情况下我们很难训练处如此高的准确度。

### cross validation
为了确保确保我们不是运气好才得到这么高的准确度，我们还可以对模型进行交叉检验(cross validation)，进一步验证模型的归纳能力。

### grid search
每一个机器学习算法都有多个超参数(hyper paramter)，不同的参数可能会得到不同的结果，我们可以用`for`循环去找出最佳的参数，但sklearn给我提供了非常好的工具——网格搜索(grid search)——帮助我们找最佳参数。


# 总结
> 本文只简单举例了一些sklearn在机器学习中的基本操作，旨在说明sklearn降低了机器学习应用的门槛，让更多学习者了解到机器学习的效果。在实际应用中，sklearn还有更多有用的工具，读者可详细阅读[sklearn API文档](https://scikit-learn.org/stable/modules/classes.html)做进一步的学习