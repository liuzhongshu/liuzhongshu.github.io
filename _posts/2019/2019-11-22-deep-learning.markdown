---
published: true
title: 深度学习
layout: post
---

## 人工智能
智能本身的定义是很模糊，图灵测试给出了一个实际判定人工智能的方法，但这个方法本身也存疑，有点哲学化了。

机器学习致力于开发一个算法，可以像人类一样学习。这个算法的特征就是：基于经验随时间不断改善性能，这和常规算法是非常不一样的。现实场景中很多问题不能实现普通算法，只能用机器学习来解决。

## 深度学习

深度学习的技术很早就有，以前叫做多层感知机。随着BP算法和GPU加速大大提高深度网络的训练效率，近年逐渐热起来，神经网络很擅长处理输入为语音、图像、视频类的场景，并且逐渐扩展到很多领域，比如游戏、视觉生成等, NLP（自然语言处理）。

![](../../public/images/2019-12-04-16-18-46.png)

“深度”通常是指多层的神经网络，具体几层算深度，并没有明确定义，看一下典型的应用：

* 2012 AlexNet 8层 错误率 16.4%
* 2014 VGG 19层 错误率 7.3%
* 2015 ResidualNet 152层 错误率 3.57% - 超过人类的识别率
* [这里](http://playground.tensorflow.org/) 有一个在线演示深度学习的例子，可以图形化的看到完整的训练过程中Loss的变化过程，甚至图形化的方式看到每个神经元起的作用。

为什么要深度，而不是仅仅增加单层神经元数量？理论上，一个二层的神经网络可以拟合任意函数，但是如果深度不够，拟合任意函数需要的神经元太多了（数学上可以证明），所以深度是兼具拟合度和训练效率的考虑。


## 前提
先安装Anaconda 3.7，之所以选择3.7，是因为将来很多库已经不支持2.x的版本。先安装Anaconda包括了很多基础库：

* NumPy 提供了大量数组和矩阵运算
* SciPy 提供了很多数值算法
* Matplotlib 提供了绘图工具

上面内置的三个库，构成了python替代matlab的基础。Matplotlib可以使用两个导入方式，通常交互环境下可以用pylab。

    pylab的show()缺省是阻塞式的，调测的时候，可以先执行pylab.ion()打开交互模式，再plot/show的时候就不阻塞了，ioff()可以关闭交互模式。

Anaconda还内置了jupyter notebook，可以方便的集成代码和文档。除此以外，对于机器学习，需要安装下面的库：

* [PyTorch](https://pytorch.org/get-started/locally/) facebook开源的深度学习框架，可以通过命令行在Anaconda下一条命令安装。

PyTorch目前的GPU加速只支持CUDA，所以，最好购买计算机时选择Nvidia显卡，即使最低端的MX150，也可以很好的支持GPU加速，只是因为显存不足，在训练大数据量时会有不足，但对于开发用是不错的选择。

## 线性回归

线性回归可以看作机器学习的基础，它的代价函数通常为RMSE - 均方根误差，也可以用MSE-均方误差, 下面是一个最小二乘法(Least Squares)实现的回归例子（最小二乘可以直接用方程推导）。

```
import pylab
import numpy
pylab.ion()

x = numpy.linspace(-1,1,100)
signal = 2 + x + 2 * x * x
noise = numpy.random.normal(0, 0.1, 100)
y = signal + noise
x_train = x[0:80]
y_train = y[0:80]

degree = 9
X_train = numpy.column_stack([numpy.power(x_train,i) for i in xrange(0,degree)])
model = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(),X_train)),X_train.transpose()),y_train)
pylab.plot(x,y,'g')
pylab.xlabel("x")
pylab.ylabel("y")
predicted = numpy.dot(model, [numpy.power(x,i) for i in xrange(0,degree)])
pylab.plot(x, predicted,'r')
pylab.legend(["Actual", "Predicted"], loc = 2)
train_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80])))
test_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[80:] - predicted[80:], y[80:] - predicted[80:])))
```

解释一下numpy的几个api：

* numpy.column_stack
* numpy.dot  点积/点乘/内积

这是一个典型的欠拟合，将degree改为3和9，分别可以得到2阶和8阶多项式拟合的结果，结果分别对应比较好的拟合和过拟合，因为实验数据解来自于二阶多项式加噪声，可以理解为什么二阶拟合最佳，但实际情况中很难知道实验数据是几阶的，所以调节模型的阶数并不容易。另外，如果学习数据量很小，也不能用过高的阶数的拟合，比如50个学习数据，如果用50阶来拟合，就100%过拟合，没有任何意义。

为了解决过拟合问题，一种方法当然是提高训练集数量，另一个方法是引入正则化概念，在模型上引入一个正则项，用于惩罚过多的阶数，正则后的模型可以更好的防止过拟合。

## 神经网络

神经网络中的基本元素是神经元，神经元做的事情很简单，一个线性变换加一个激活变换（非线性）。

![](../../public/images/2019-12-02-16-18-13.png)

之所以加激活函数，这个是神经网络的灵魂，没这个函数，再多层的神经网络叠加后本质上还是只是一个线性变换，当然为了梯度下降算法能够应用，我们需要激活函数可以连续微分。激活函数到目前为止，常用的有：

![](../../public/images/2019-12-02-16-21-27.png)

之所以最早使用sigmoid函数，因为它的输出是0~1，符合很多分类预测问题概率输出的需要。但实际情况中，sigmoid并非最佳。

* sigmoid 现在常用在二元分类网络的输出层，输出0~1间的概率分布。
* relu函数的梯度更大，更适合梯段下降算法，常用于hidden layer，tanh也还不错。

softmax layer 常用在多分类网络的输出层，它很简单，把上一层的每个分类的可能性输出转换为0~1的概率。

训练神经网络的方法通常为梯度下降法, Loss函数可以用最大似然估计来求解，可以得到三个函数（求解过程略）：

* Binary Cross entropy 
* Cross entropy 
* Squared loss function

## FNN

FNN是前馈神经网络

卷积
线性回归
非线性函数logistics, Softmax, 
价值函数
反向传播算法

## CNN
针对图像类应用，FNN网络训练效率不高，考虑下面几个特性，CNN网络做了特别设计：

* 图像的pattern通常比整个图像小。
* 图像的pattern可能出现在很多地方。
* 图像可以被缩小，并不影响识别。

![](../../public/images/2019-12-03-17-10-53.png)

* Convolution针对上面pattern的两个特点优化
* Max pooling优化上面的缩放特性

CNN的典型应用：
* deep dream 让图片中的“特征”更明显
* deep style 让一张照片具备另一张照片的style
* 著名的alpha go，之所以用CNN，大概是因为围棋也具有上述图像的前两个特征，但除去了Max pooling。

## RNN

## 参考
* https://pytorch.org/tutorials/
* https://github.com/Atcold/PyTorch-Deep-Learning-Minicourse
