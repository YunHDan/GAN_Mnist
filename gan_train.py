import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from Model import generator
from Model import discriminator

import os

if not os.path.exists('gan_train.py'):  # 报错中间结果
    os.mkdir('gan_train.py')


def to_img(x):  # 将结果的-zero.5~zero.5变为0~1保存图片
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 96
num_epoch = 200
z_dimension = 100

# 数据预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),  # 图像数据转换成了张量，并且归一化到了[zero,1]。
    transforms.Normalize([0.5], [0.5])  # 这一句的实际结果是将[zero，1]的张量归一化到[-1, 1]上。前面的（zero.5）均值， 后面(zero.5)标准差，
])
# MNIST数据集
mnist = datasets.MNIST(
    root='./data', train=True, transform=img_transform, download=True)
# 数据集加载器
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True)

D = discriminator()  # 创建生成器
G = generator()  # 创建判别器
if torch.cuda.is_available():  # 放入GPU
    D = D.cuda()
    G = G.cuda()

criterion = nn.BCELoss()  # BCELoss 因为可以当成是一个分类任务，如果后面不加Sigmod就用BCEWithLogitsLoss
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)  # 优化器
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)  # 优化器

# 开始训练
for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):  # img[96,1,28,28]
        G.train()
        num_img = img.size(0)  # num_img=batchsize
        # =================train discriminator
        img = img.view(num_img, -1)  # 把图片拉平,为了输入判别器 [96,784]
        real_img = img.cuda()  # 装进cuda，真实图片

        real_label = torch.ones(num_img).reshape(num_img, 1).cuda()  # 希望判别器对real_img输出为1 [96,1]
        fake_label = torch.zeros(num_img).reshape(num_img, 1).cuda()  # 希望判别器对fake_img输出为0  [96,1]

        # 先训练鉴别器
        # 计算真实图片的loss
        real_out = D(real_img)  # 将真实图片输入鉴别器 [96,1]
        d_loss_real = criterion(real_out, real_label)  # 希望real_out越接近1越好 [1]
        real_scores = real_out  # 后面print用的

        # 计算生成图片的loss
        z = torch.randn(num_img, z_dimension).cuda()  # 创建一个100维度的随机噪声作为生成器的输入 [96,1]
        #   这个z维度和生成器第一个Linear第一个参数一致
        # 避免计算G的梯度
        fake_img = G(z).detach()  # 生成伪造图片 [96,748]
        fake_out = D(fake_img)  # 给判别器判断生成的好不好 [96,1]

        d_loss_fake = criterion(fake_out, fake_label)  # 希望判别器给fake_out越接近0越好 [1]
        fake_scores = fake_out  # 后面print用的

        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 训练生成器
        # 计算生成图片的loss
        z = torch.randn(num_img, z_dimension).cuda()  # 生成随机噪声 [96,100]

        fake_img = G(z)  # 生成器伪造图像 [96,784]
        output = D(fake_img)  # 将伪造图像给判别器判断真伪 [96,1]
        g_loss = criterion(output, real_label)  # 生成器希望判别器给的值越接近1越好 [1]

        # 更新生成器
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f'Epoch [{epoch}/{num_epoch}], d_loss: {d_loss.cpu().detach():.6f}, g_loss: {g_loss.cpu().detach():.6f}',
                f'D real: {real_scores.cpu().detach().mean():.6f}, D fake: {fake_scores.cpu().detach().mean():.6f}')
    if epoch == 0:  # 保存图片
        real_images = to_img(real_img.detach().cpu())
        save_image(real_images, './img_gan/real_images.png')

    fake_images = to_img(fake_img.detach().cpu())
    save_image(fake_images, f'./img_gan/fake_images-{epoch + 1}.png')

    G.eval()
    with torch.no_grad():
        new_z = torch.randn(batch_size, 100).cuda()
        test_img = G(new_z)
        print(test_img.shape)
        test_img = to_img(test_img.detach().cpu())
        test_path = f'./test_result/the_{epoch}.png'
        save_image(test_img, test_path)

# 保存模型
torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')
