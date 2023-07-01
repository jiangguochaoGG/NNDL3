# 对比ResNet-18与Transformers在CIFAR-100数据集上的差异

[CIFAR-100数据库](https://www.cs.toronto.edu/~kriz/cifar.html)，需要使用库`timm`。

## 使用

1. 执行`python check_model.py`来展示模型信息（需要库`torchinfo`）

2. 执行`python train_model.py [model]`来训练模型`model`。`model`可以是`resnet`、`vit26`或者`vit11`

