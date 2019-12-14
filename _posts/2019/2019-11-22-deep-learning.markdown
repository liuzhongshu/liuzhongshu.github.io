---
published: true
title: 深度学习
layout: post
---

## 人工智能
智能本身的定义是很模糊，图灵测试给出了一个实际判定人工智能的方法，但这个方法本身也存疑，有点哲学化了。

机器学习本质上也是一个算法，但和传统算法完全不一样，传统的计算机算法，是有了输入，通过设计好的算法，产生输出，但很多现实问题很复杂，没办法写出算法，但是可以有一些例子（经验数据），机器学习就是在经验数据上推导出算法的算法，再用推导出的算法就求解新的输入数据。机器学习算法的严谨定义就是：基于经验随时间不断改善性能。

## 深度学习

深度学习的技术很早就有，以前叫做多层感知机。随着BP算法和GPU加速大大提高深度网络的训练效率，近年逐渐热起来，神经网络很擅长处理输入为语音、图像、视频类的场景，并且逐渐扩展到很多领域，比如游戏、视觉生成等, NLP（自然语言处理）。

![](../../public/images/2019-12-04-16-18-46.png)

“深度”通常是指多层的神经网络，具体几层算深度，并没有明确定义，看一下深度学习的发展历史和典型的应用：
* 1950 MLP多层感知机诞生，其实就是现代深度神经网络的原型，在80年代曾经很流行。
* 1998 LENET 第一个CNN模型，提出很多影响后世的思想，只是因为当时算力不够，此后深度学习陷入低潮。
* 2012 AlexNet 8层 ImageNet错误率16.4% - 重新开启深度学习大潮
* 2014 VGG 19层 ImageNet错误率7.3%
* 2015 ResidualNet 152层 ImageNet错误率3.57% - 超过人类的识别率

[这里](http://playground.tensorflow.org/) 有一个在线演示深度学习的例子，可以图形化的看到完整的训练过程中Loss的变化过程，甚至图形化的方式看到每个神经元起的作用。

数学上可以证明，一个二层的神经网络可以拟合任意函数，那为什么要深度，而不是仅仅增加单层神经元数量？ 因为数学上同时可以证明，如果深度不够，拟合任意函数需要的神经元太多了，所以深度是兼具拟合度和训练效率的考虑。

## 前提
先安装Anaconda 3.7，之所以选择3.7，是因为将来很多库已经不支持2.x的版本。Anaconda包括了很多基础库：

* NumPy 提供了大量数组和矩阵运算
* SciPy 提供了很多数值算法
* Matplotlib 提供了绘图工具

上面内置的三个库，构成了python替代matlab的基础。Matplotlib可以使用两个导入方式，通常交互环境下可以用pylab。

    pylab的show()缺省是阻塞式的，调测的时候，可以先执行pylab.ion()打开交互模式，再plot/show的时候就不阻塞了，ioff()可以关闭交互模式。

Anaconda还内置了jupyter notebook，可以方便的集成代码和文档。除此以外，对于深度学习，需要安装pyTorch。

## pyTorch
[PyTorch](https://pytorch.org/get-started/locally/) 是facebook开源的深度学习框架，在学术界基本一统江湖。可以通过命令行在Anaconda下一条命令安装。
```
PyTorch目前的GPU加速只支持CUDA，所以，最好购买计算机时选择Nvidia显卡，即使最低端的显卡，相对CPU也有数倍的速度提升，如果要训练大数据量需要足够的显存。
```

pyTorch的核心数据结构是Tensor，称为矢量或张量，torch.Tensor有一个重要的属性requires_grad，如果设为True，pyTorch会自动跟踪这个Tensor上的计算，在计算完成后调用.backward(), 就会在.grad属性上计算出梯度(gradients)，这就是所谓的自动微分autograd，这个特性大大简化训练代码编写。

torch.nn.Module是定义模型的基类, 在这个基类上，实现一个forward方法就可以完整的定义一个深度模型，同时nn下也封装了很多预定义好的神经元。通常在写forward方法时，还需要用到大量的torch.nn.functional里的数学方法，比如max_pool2d、relu等。



```
很多人说到的调参实际上指的是网络结构上的参数，这些是不能通过训练得到的参数，而不是那些神经元上的可训练出的参数。
```

## 图像识别数据集

在DL历史上，MNIST是非常著名的一套手写数字识别图像集，但是它过于简单了，除此以外，还有[fashion-mnist](https://github.com/zalandoresearch/fashion-mnist), CIFAR10等数据集，[这里](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)有一个各个数据集及相应识别率的汇总页面。

fashion-mnist，60000张10类，28*28黑白图像，相对MNIST更难一些，使用pyTorch可以方便的加载fashion-mnist，如果加载太慢，可以先从github下载下来，放到对应的目录下，一共4个文件。

```
import torchvision
train_data = torchvision.datasets.FashionMNIST(root='~/ai/mnist', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root='~/ai/mnist', train=False, download=True, transform=torchvision.transforms.ToTensor())
```

CIFAR10，60000张10类，32*32彩色图像, 目前最高识别率大概在96%，想更难的话，可以试试100分类的CIFAR100，识别率大概到了75%。

```
import torchvision
train_data = torchvision.datasets.CIFAR10(root='~/ai/cifar10', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='~/ai/cifar10', train=False, download=True, transform=torchvision.transforms.ToTensor())
```

有了dataset，还需要一个dataloader，dataloader可以从dataset中加载批量数据，批量(batch_size)可以让loader一次加载多个数据，这样不仅通过GPU训练更高效，loader遍历一遍数据也变快。loader也可以通过num_workers支持并行加载，但在windows下只能设置为0。

```
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True, num_workers=0)
```

## 多元分类
针对图像识别这种多元分类问题，最简单的深度网络就是SoftMax，SoftMax是在线性回归基础上发展出的分类算法，原理很简单

* 因为有多元，原有的线性回归输出层要变成N元，每个表示一个分类的置信度。
* 绝对置信度不好理解，也不容易和训练标签对应，所以将每个置信度理解为0~1的相对概率。

相应的，对应多元分类，传统线性回归的损失函数也不太合适，因为平方误差函数过于严格，而softmax输出值虽然有N个，但只有最大值有意义，通常我们使用交叉熵损失函数。

![](../../public/images/2019-12-13-10-16-33.png)

很明显，q个输出里y值只有一个为1，所以前面用y来乘，后面取log，是让对应的输出为1时, 损失函数最小，下面是一个简单的多元分类训练程序。

```
import time
import torch
import torchvision

train_data = torchvision.datasets.FashionMNIST(root='~/ai/mnist', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root='~/ai/mnist', train=False, download=True, transform=torchvision.transforms.ToTensor())
# num_workers must be 0 under windows, batch_size is key for loader performance
train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True, num_workers=0)

class Net(torch.nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.fc = torch.nn.Linear(28 * 28, 10)
  def forward(self, x):
      x = x.view(x.shape[0], -1)
      x = self.fc(x)
      return x

net = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
num_epochs = 5
print("%d: start training..." % (time.time()))

for epoch in range(num_epochs):
  train_loss, train_acc, n = 0.0, 0.0, 0
  for X, y in train_loader:
    optimizer.zero_grad()

    y_hat = net(X)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()

    y = y.type(torch.float32)
    train_loss += loss.item()
    train_acc += torch.sum((torch.argmax(y_hat, dim=1).type(torch.FloatTensor) == y).detach()).float()
    n += list(y.size())[0]
  print('%d: epoch %d, loss %.4f, train acc %.3f' % (time.time(), epoch + 1, train_loss / n, train_acc / n))

```

上述代码使用CPU训练了5轮，lr可以控制收敛的速度，输出如下：

```
1576303662: start training...
1576303675: epoch 1, loss 0.0370, train acc 0.766
1576303689: epoch 2, loss 0.0274, train acc 0.821
1576303703: epoch 3, loss 0.0253, train acc 0.831
1576303717: epoch 4, loss 0.0243, train acc 0.837
1576303731: epoch 5, loss 0.0235, train acc 0.842
```

如果一直训练下去，大概到86%就最高了，这个网络只有一层输出，没有hidden layer。

## FNN

FNN是前馈神经网络, 其实就是多层感知机MLP换一个名字，pyTorch来实现一个含有一个隐层的FNN

```
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyper-parameters 
input_size = 28 * 28
num_classes = 10
hidden_size = 500
num_epochs = 2
batch_size = 100
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = torchvision.datasets.FashionMNIST(root='~/ai/mnist', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root='~/ai/mnist', train=False, download=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)        
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('{:0.0f}: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(time.time(), epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model, In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
```    

这里我们使用了Adam训练算法，大大提高收敛速度（即使lr很小），结果和上面的单层多元分类网络差不多：

```
1576308516: Epoch [1/2], Step [100/600], Loss: 0.6881
1576308519: Epoch [1/2], Step [200/600], Loss: 0.4680
1576308522: Epoch [1/2], Step [300/600], Loss: 0.4607
1576308525: Epoch [1/2], Step [400/600], Loss: 0.4143
1576308527: Epoch [1/2], Step [500/600], Loss: 0.4079
1576308530: Epoch [1/2], Step [600/600], Loss: 0.3841
1576308533: Epoch [2/2], Step [100/600], Loss: 0.3728
1576308536: Epoch [2/2], Step [200/600], Loss: 0.3185
1576308539: Epoch [2/2], Step [300/600], Loss: 0.3990
1576308542: Epoch [2/2], Step [400/600], Loss: 0.4038
1576308545: Epoch [2/2], Step [500/600], Loss: 0.2982
1576308549: Epoch [2/2], Step [600/600], Loss: 0.3476
Accuracy of the network on the 10000 test images: 86.64 %
```

提醒下，上述代码已经支持GPU，但我的计算机没有GPU，所以速度上比较慢。

## CNN
CNN就是含卷积层的神经网络，针对图像类应用，FNN网络训练效率不高，考虑下面几个特性，CNN网络做了特别设计：

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

## RNN 与 NLP
语言模型（language model）是自然语言处理的重要技术。自然语言处理中最常见的数据是文本数据。我们可以把一段自然语言文本看作一段离散的时间序列。假设一段长度为T的文本中的词依次为w1, w2... wt，语言模型将计算该序列的概率：

P(w1, w2, ... wt)

语言模型可用于提升语音识别和机器翻译的性能。例如，在语音识别中，给定一段“厨房里食油用完了”的语音，有可能会输出“厨房里食油用完了”和“厨房里石油用完了”这两个读音完全一样的文本序列。如果语言模型判断出前者的概率大于后者的概率，我们就可以根据相同读音的语音输出“厨房里食油用完了”的文本序列。在机器翻译中，如果对英文“you go first”逐词翻译成中文的话，可能得到“你走先”“你先走”等排列方式的文本序列。如果语言模型判断出“你先走”的概率大于其他排列方式的文本序列的概率，我们就可以把“you go first”翻译成“你先走”。

P(w1, w2, ... wt)的概率随着t的增长，复杂度会指数级增长，所以通过马尔科夫假设来简化模型，即每个词只和前面的n个词相关，n作为一个模型准确率和复杂度的权衡。

## 参考

按顺序给出深度学习的一些好的参考资料和书籍。

* [Pytorch的官方教程](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
* [Dive into DL的pytorch版本](https://github.com/ShusenTang/Dive-into-DL-PyTorch), 这本书中英版本有明显不同，英文版本内容更丰富一些。
* [基于纯代码的Pytorch教程](https://github.com/yunjey/pytorch-tutorial), 代码很整洁，值得学习。

* [Pytorch的图片转换实现](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* https://github.com/huggingface/transformers

