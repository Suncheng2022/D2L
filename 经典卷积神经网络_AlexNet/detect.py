import argparse
import torch

from torchvision import transforms
from model import AlexNet
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='detect of AlexNet by sc. 2022.12.07')
    parser.add_argument('--model_path', default=r'./runs/model_best.pth', metavar='PATH', help='选择进行前向推断的模型')
    parser.add_argument('--source', default='./images/img.png', type=str, help='执行分类的图像文件')

    opt = parser.parse_args()
    print(f'Hyper param:{opt}')

    device = torch.device(f'cuda:0') if torch.cuda.device_count() else torch.device('cpu')
    print(f'classificating on {device}')
    model = AlexNet()
    model.to(device)
    model.load_state_dict(torch.load(opt.model_path))

    img = Image.open(opt.source).convert('L')
    trans = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((224, 224))])
    img = model(trans(img).reshape(-1, 1, 224, 224).to(device))

    fashion_mnist = {0: 'T恤', 1: '裤子', 2: '套衫', 3: '裙子', 4: '外套',
                     5: '凉鞋', 6: '汗衫', 7: '运动鞋', 8: '包', 9: '靴子'}
    score = torch.softmax(img, dim=1)
    cls = torch.argmax(score, dim=1)
    print(f'识别结果:{fashion_mnist[cls.item()]}, 置信度:{score[0, cls.item()]}')


if __name__ == '__main__':
    main()
