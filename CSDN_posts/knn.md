# 【机器学习应用】【Python】K近邻（KNN）

## mglearn
在开始之前再介绍一个机器学习算法的工具包`mglearn`，是《Introduction to Machine Learning with Python》整本书使用的一个工具包，完整代码参考[github](https://github.com/amueller/introduction_to_ml_with_python)

安装mglearn参考[set up](https://github.com/amueller/introduction_to_ml_with_python/tree/master#setup)
```python
# 终端安装
pip install mglearn

# Jupyter Notebook安装
!pip install mglearn
```

## K近邻简介
K近邻算法是一种适用于回归和分类的机器学习算法，它的基本原理是根据距离某一个数据点最近的k个数据的类别对其进行分类。例如，当k=1时，对于一个数据点，KNN算法会将其分类为离它最近的第一个数据的类别。

![knn1](pics/knn1.png)

通常计算数据点和周围数据的距离方法有三种：
1. 欧氏距离（Euclidean distance）
2. 曼哈顿距离（Manhattan distance）
3. 明可夫斯基距离（Minkowski distance）

## 如何使用KNN
以乳腺癌数据集`load_breast_cancer`为示例：
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer_data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, random_state=42)

# 创建一个KNN模型
model = KNeighborsClassifier(n_neighbors=3)

# 拟合模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 查看准确度
print("Accuracy: {:.2f}".format(model.score(X_test, y_test)))
```
输出：
```
Accuracy: 0.93
```
### 到底多少个邻居才能确定类别
俗话说，类以物聚，人以群分，但是到底要了解多少一个人身边的邻居，才能更好的了解这个人呢？这时候就需要我们调整k值来测试一下。

k值，也就是我们的近邻数(`n_neighbors`)。目的是告诉KNN算法，在做预测时需要参考一个数据点周围的k个数据。

```python
training_accuracy = []
test_accuracy = []

neighbor_settings = range(1, 11)

for neighbor in neighbor_settings:
    model = KNeighborsClassifier(n_neighbors=neighbor)
    model.fit(X_train, y_train)
    training_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))

plt.figure(figsize=(8,6))
plt.plot(neighbor_settings, training_accuracy, label="training accuracy")
plt.plot(neighbor_settings, test_accuracy, label="testing accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
```
输出：
![knn2](pics/knn2.png)

分析一下结果可以发现，近邻数为5时预测准确度已经达到96.5%左右，之后有一个小幅度的下降。在近邻数为10时准确度有明显的提升，但是训练准确度却有所下降。  
目前来看呢，近邻数为5较为合适。如果进一步的实验，我们可以：
1. 交叉检验；
2. 尝试另外一种距离算法
3. 增加更多数据

## 所以knn到底怎么分类的呢？
光看数据的话，我们还不能直观的理解K近邻的分类方式，或者它的依据是什么。那么我们可以将它的决策结果可视化，帮助我们分析，也就是画出它的决策分界线(Decision Boundary)，

```

```
通过上面几张图我们可以发现，近邻数越多，分界线可能会更平滑。

这看起来是个不错的可视化，但实际上我们只能画出2维或3维，因为更高维度的分界线，已经不是我们人脑可以理解的了...

## 最后
> 一般来说，K近邻是机器学习中最简单的算法之一，也基本适用于生活中的许多问题。所以当我们想要使用机器学习算法做预测时，不妨从尝试一下K近邻算法。

另外，画出一个漂亮的Decision Boundary能够帮助我们更好的分析算法的结果，更多的参考如下：  
* [Beautiful Plots: The Decision Boundary](https://www.tvhahn.com/posts/beautiful-plots-decision-boundary/)
* [Visualizing Graph k-NN Decision Boundaries with Matplotlib ](https://saturncloud.io/blog/visualizing-graph-knn-decision-boundaries-with-matplotlib/)
