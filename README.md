# SimCLR论文复现
[参考网址1](https://blog.csdn.net/qq_43027065/article/details/118657728?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167625610416800186517588%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167625610416800186517588&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-3-118657728-null-null.142^v73^pc_search_v2,201^v4^add_ask,239^v1^control&utm_term=SimCLR%E5%A4%8D%E7%8E%B0&spm=1018.2226.3001.4187)
SimCLR图像分类——pytorch复现

[参考网址2](http://t.csdn.cn/giaE2)
SimCLR框架解析

[餐卡网址3](https://blog.csdn.net/qq_46006468/article/details/126066506)
自监督学习和对比学习的理解

[数据集网址](http://www.cs.toronto.edu/~kriz/cifar.html)
CIFAR数据集

## Introduction

### From ChatGPT

在本次学习中，选择从ChatGPT学习一些内容，感觉其回答很令人满意。

以下是结果展示：

**Q：你有读过这篇论文吗《A Simple Framework for Contrastive Learning of Visual Representations》**

A：是的，我熟悉这篇论文。《A Simple Framework for Contrastive Learning of Visual Representations》是一篇由 Google Brain 团队提出的，发表在ICML 2020上的论文，提出了一个简单而有效的对比学习框架，用于学习视觉表示。

这篇论文的核心思想是，通过对数据集中不同的视觉样本进行比较，学习出一个好的视觉表示。这种*对比学习*的方式可以通过构造一个二元组，将两个来自不同图像的视觉样本作为输入，然后将这两个样本映射到一个共享的特征空间中，并且通过优化损失函数，使得同类样本之间的距离尽可能小，异类样本之间的距离尽可能大。

这篇论文提出的方法是一种*无监督学习的方法*，可以在没有标签的情况下学习到视觉表示。实验结果表明，这种方法可以在许多视觉任务上取得良好的表现，包括图像分类、目标检测等。

总之，这篇论文提出的对比学习框架为计算机视觉领域提供了一种有效的视觉表示学习方法，对于深度学习和计算机视觉研究具有一定的参考价值。

**Q:在SimCLR这种自监督学习算法中，有哪些很重要的地方**

A：SimCLR是一种目前在自监督学习中非常成功的方法，它在ImageNet等数据集上取得了很好的表现。以下是SimCLR中几个重要的方面：

1. 数据增强：数据增强是SimCLR中非常重要的部分。为了让模型从未标记的数据中学习到有用的表示，SimCLR使用了多种数据增强策略，包括裁剪、翻转、旋转、颜色扰动等。
这些增强策略可以增加数据集的多样性，从而帮助模型学习到更加鲁棒和通用的特征表示。 
2. 对比损失函数：SimCLR使用对比损失函数来鼓励模型学习到在不同变换下具有一致性的特征表示。
该损失函数首先将每个图像扩充为两个视角，并将它们的特征表示向量映射到同一维度空间中。
然后，使用余弦相似度来度量这两个特征表示向量之间的相似性。
模型被要求最小化同类样本之间的相似度，同时最大化不同类样本之间的相似度。
这个损失函数强制模型在学习特征表示时关注样本间的差异，从而提高模型的鲁棒性和泛化性能。 
3. 大型神经网络：SimCLR通常使用非常大的神经网络，例如ResNet-50等，以获得更好的特征表示。
同时，为了提高训练效率，SimCLR使用了分布式训练技术，并使用多个GPU对网络进行训练。 
4. 调整超参数：SimCLR中有多个超参数需要调整，包括数据增强策略、批量大小、学习率、优化器等等。
调整这些超参数通常需要进行一些实验和测试，以找到最佳的参数配置。

综上所述，数据增强、对比损失函数、大型神经网络和超参数调整是SimCLR中非常重要的方面。这些技术的使用可以帮助模型学习到更加鲁棒和通用的特征表示，从而提高模型在自监督学习任务中的表现。

## 复现Project

### 原理

SimCLR是一种对比学习网络，包括**无监督学习和有监督学习**两个部分。
通过对比学习实现的自监督学习。

#### 来自[1]中的介绍：

在特征提取中采用ResNet50网络，主要做法是：将输入层进行更改，并去掉池化层及全连接层。
之后将特征图平坦化，并依次进行全连接、批次标准化、ReLU激活、全连接，得到输出特征。
有监督学习网络使用无监督学习网络的特征提取层及参数，之后由一个全连接层得到分类输出。
在第一阶段先进行无监督学习，对输入图像进行两次随机图像增强，即由一幅图像得到两个随机处理过后的图像，依次放入网络进行训练，计算损失并更新梯度。

#### 我的理解
**Four Major Components:**
1. **Data Augmentation**:
   1. random cropping
   2. random color distortions
   3. random Gaussian blur

random cropping和random color distortions的结合非常重要。

2. **Encoder** ==> f(*) ==> ResNet ==> Extract Representations
3. **Projection Head** ==> MLP :其主要作用是将输入数据（如图像、文本等）的高维特征映射到一个低维的向量空间中。

PS:在自监督学习中，Projection Head通常是与一个Encoder模型配对使用的。
Encoder模型是一个用于提取输入数据特征的深度神经网络，Projection Head则接收Encoder的输出并将其映射到低维空间。整个模型可以被认为是一个自监督学习框架，其中Encoder被训练来学习一个好的特征提取模型，而Projection Head则被训练来学习如何将这些特征转换为更具有区分度的表示。

4. **Contrastive Loss Function** ==> contrastive prediction task; 
采用的是余弦相似度来进行计算。具体讲解可以看[2]。

### 工程框架
#### net.py
搭建网络

以下训练方式（具体的任务和算法来确定自监督学习算法是无监督训练还是有监督训练）：
1. Stage One 无监督训练
2. Stage Two 有监督训练

搭建了两个网络，SimCLRStage1和SimCLRStage2两个网络，分别用于无监督学习和有监督学习。
同时定义损失函数Loss，使用sim(u,v)

#### config.py

配置文件

配置内容包括
1. GPU的调用
2. model的保存地址，以及调用地址
3. train-transform：对图片进行随机变换等；相当于Data Augmentation
4. test-transform: 对测试集进行规范化

#### loaddataset.py

数据集的加载

数据集的类别:CIFAR-10数据集，包含10个类别的RGB图片；
图片尺寸为32*32；
数据集中一共有50000张训练图片和10000张测试集图片。

#### trainstage1.py

无监督学习部分

我更倾向于说这部分是自监督学习部分，通过net.py中的SimCLR1()网络函数进行训练。
在本部分训练中得到最好的model，可以用与stage2的训练。


#### trainstage2.py

带有少量标签的学习，训练最终得到的model可以用test.py文件中，对输入的图片进行预测。

本部分一共有记录了损失函数、准确率（top1_acc,top5_acc）、混淆矩阵。

混淆矩阵为我自己添加的，目的是为了在十分类任务中更好地看出分类结果。

#### visualization.py

可视化，绘制损失函数

#### confusionMatrix.py

这部分我使用了sklearn库，去绘制混淆矩阵。

[参考网址](http://t.csdn.cn/cVDN0)
讲述很好，还有配色；方法相对比较简单


## 总结

### 复现部分

作为第一次复现的深度学习框架，具体来说框架都是来自[1]中的内容。非常感谢博主[1]的分享。

参数设置：在本次训练中我的batch_size设定在128，一共迭代了100个epoch。

我对可视化部分进行了修改，放弃使用visdom库，使用常用的matplotlib进行绘制。

### 上传存档-GitHub

作为刚入门的小白，所以想尝试将所完成的本次工程上传到GitHub上进行存档；如果[1]的作者感觉不妥，可随时联系我删除。

### 学习收获及问题

收获：

1. SimCLR框架 
2. sklearn绘制混淆矩阵
3. 使用os库去对文件进行调动

问题：

1. 网络框架的具体使用方式，比如ResNet。
2. 对于trainstage2.py中的prediction为什么取最小值
3. 没用过cv库；但不知道在研究生生涯中的作用大小。


