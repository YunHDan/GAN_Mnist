import os
import sys
import numpy as np
import torch
import argparse
import torch.utils.data
from PIL import Image
from Model import FinetuneModel
from Model import generator
from torchvision.utils import save_image

parser = argparse.ArgumentParser("GAN")
parser.add_argument('--save_path', type=str, default='./test_result')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--model', type=str, default='generator.pth')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)


def to_img(x):  # 将结果的-zero.5~zero.5变为0~1保存图片
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


def main():
    if not torch.cuda.is_available():
        print("no gpu device available")
        sys.exit(1)

    model = FinetuneModel(args.model)
    model = model.to(device=args.gpu)
    model.eval()

    z_dimension = 100

    with torch.no_grad():
        for i in range(100):
            z = torch.randn(96, z_dimension).cuda()  # 创建一个100维度的随机噪声作为生成器的输入 [96,100]
            output = model(z)
            print(output.shape)
            u_name = f'the_{i}.png'
            print(f'processing {u_name}')
            u_path = save_path + '/' + u_name
            output = to_img(output.cpu().detach())
            save_image(output, u_path)


if __name__ == '__main__':
    main()
