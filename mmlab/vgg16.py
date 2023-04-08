import torch
import os
import cv2
import numpy as np
import logging
from mmpretrain import inference_model, get_model
from torch import nn
from PIL import Image
from torchvision import transforms
from collections import OrderedDict

os.environ['TORCH_HOME'] = '../data/pretrain_weights/'


class MyVGG16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(*[
            # ConvModule_0
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(inplace=True),

            # ConvModule_1
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ConvModule_3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5),
            nn.ReLU(inplace=True),

            # ConvModule_4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ConvModule_6
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5),
            nn.ReLU(inplace=True),

            # ConvModule_7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5),
            nn.ReLU(inplace=True),

            # ConvModule_8
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ConvModule_10
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-5),
            nn.ReLU(inplace=True),

            # ConvModule_11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-5),
            nn.ReLU(inplace=True),

            # ConvModule_12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ConvModule_14
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-5),
            nn.ReLU(inplace=True),

            # ConvModule_15
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-5),
            nn.ReLU(inplace=True),     

            # ConvModule_16
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),    

            nn.Flatten(),
            # classifier
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),

            # 删除
            nn.Dropout(),
            nn.Linear(4096, 1000, bias=True)
        ])

    def __call__(self, X):
        """ 前向传播 """
        X = self.model(X)
        return X
    

if __name__ == '__main__':
    device = torch.device(f'cuda:0') if torch.cuda.device_count() else torch.device('cpu')

    mmVGG16 = get_model('vgg16bn_8xb32_in1k', pretrained=True, device='cpu')
    # print(mmVGG16.state_dict().keys())
    vgg = MyVGG16()
    # print('---------------\n', vgg.state_dict().keys())

    image = Image.open('images/catdog.jpg')
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    img = trans(image).unsqueeze(0)
    
    mmVGG16_params = iter(mmVGG16.state_dict().values())
    weights = OrderedDict()
    for k in vgg.state_dict().keys():
        weights[k] = next(mmVGG16_params)
    logging.info('copy weights done.')
    vgg.load_state_dict(weights)
    vgg.eval()      # 开启eval()后 二者预测结果相同了！
    print(f'{vgg(img).argmax()}\n'
          f'{mmVGG16(img).argmax()}')


    # 测试对摄像头画面分类
    # video = cv2.VideoCapture(0)
    # model = get_model('vgg16bn_8xb32_in1k', pretrained=True)
    # while True:
    #     ret, frame = video.read()   # 图像是numpy [H W C]；是否打开摄像头、帧图像
    #     if not ret:
    #         print(f'camera failed!')
    #         break
    #     cv2.imshow("my camera", frame)
    #     pred = inference_model(model, frame)
    #     print(f'----------------------> {pred["pred_class"]} {pred["pred_score"]}')
    #     if cv2.waitKey(1000) == 27:    # esc键
    #         print(f'user exit!')
    #         break
        