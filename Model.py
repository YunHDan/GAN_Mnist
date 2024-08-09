import torch
from torch import nn


# 判别器 判别图片是不是来自MNIST数据集
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),  # 784=28*28
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
            #   sigmoid输出这个生成器是或不是原图片，是二分类
        )

    def forward(self, x):
        x = self.dis(x)
        return x


# 生成器 生成伪造的MNIST数据集
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),  # 输入为100维的随机噪声
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            #   生成器输出的特征维和正常图片一样，这是一个可参考的点
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x


class FinetuneModel(nn.Module):
    def __init__(self, weights):
        super(FinetuneModel, self).__init__()
        self.G = generator()
        base_weights = torch.load(weights)

        model_parameters = dict(self.G.named_parameters())
        #   不是对model进行named_parameters，而是对model里面的具体网络进行named_parameters取出参数，否则取出的是model冗余的参数去测试
        pretrained_weights = {k: v for k, v in base_weights.items() if k in model_parameters}

        new_state_dict = {k: pretrained_weights[k] for k in model_parameters.keys()}
        self.G.load_state_dict(new_state_dict)

    def forward(self, input):
        output = self.G(input)
        return output
