# 【机器学习应用】【Python】模型评估（2）—— 网格搜索 Grid Search

> 机器学习算法都有不同的超参数，通过调整超参找到最优的模型，几乎是每一个用机器学习算法解决问题的必经步骤。

要找到最佳模型，除了`for`循环以外，还可以使用一个实用的方法——**网格搜索(Grid Search)**

网格搜索其实就是给模型提供一系列我们想要测试的超参数，然后根据不同超参数创建不同模型，并检验它们的准确度。

例如，我想要测试从1到10k近邻数的模型准确度：`n_neighbors=range(1,10)`

如果只调整一个超参数，使用`for`循环还比较简单，但当我们要调整两个以上的参数时，就会有nxm个组合需要我们去尝试，也就形成了一个网格。

以调整SVM的C值和gamma值为例：

*这里添加一个表格*

## 如何使用网格搜索
当然，在测试不同超参数时我们会面对同样的问题，那就是训练集的偶然性。为了避免偶然性影响，使模型准确度更加客观，通常我们需要网格搜索和交叉检验结合使用。

sklearn就给我提供了一个强有力的帮手：`GridSearchCV`

### 网格搜索找近邻数(k值) -- KNN


### 网格搜索找C值和gamma -- SVM