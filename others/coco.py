from pycocotools.coco import COCO
from tqdm import tqdm
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pylab
from pathlib import Path
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


# dataDir = './data'
# annFile_val = os.path.join(dataDir, 'captions_val2017.json')
# coco_caps = COCO(annFile_val)

# 还不会使用cocoAPI 提取标注带有bicycle的图片
# imgId2filename = {}     # 中介变量
# for id, info in coco_caps.imgs.items():
#     imgId2filename[id] = info['file_name']
#
# imgId2caption = {}
# for id, captions in coco_caps.imgToAnns.items():
#     img_caps = []
#     for cap in captions:
#         img_caps.append(cap['caption'])
#     imgId2caption[id] = img_caps
#
# det_imgs = set()
# bicycle_caps = open(r'e:\迅雷下载\bicycle.txt', 'w')
# for id, captions in imgId2caption.items():
#     for cap in captions:
#         if 'bicycle' in cap:
#             det_imgs.add(imgId2filename[id])
#             bicycle_caps.write(f'{imgId2filename[id]}\t{"    ".join([tmp for tmp in captions])}\n')
#             break
# bicycle_caps.close()
# print(det_imgs)
# for img in tqdm(det_imgs):
#     os.makedirs(r'e:\迅雷下载\bicycle', exist_ok=True)
#     shutil.copy(os.path.join(r'E:\迅雷下载\val2017', img), os.path.join(r'e:\迅雷下载\bicycle', img))


# 学习使用CoCoAPI
"""
2022.12.22
将coco数据集文本标注带有"bicycle"的图片和文本挑选出来，尝试训练一下
train&dev暂时都使用5条文本，验证过程似乎对val集5条做了处理
file_ls要把路径写上，暂时从Lenovo电脑上把名单上的图片复制出来
应检查captions是否均为5条
发现空行！！！--某1个标注上有好多空行所导致
"""
def get_by_class(file_path, keyword):
    """
    将file_path中带有关键字keyword的图片路径、文本标注分别写在file_ls、file_caps上
    将图片复制到同目录下的train_keyword、val_keyword下
    :param file_path: coco数据集的.json标注文件
    :param keyword: 我们要找标注中带有keyword关键字的条目
    :return:
    """
    coco_caps = COCO(file_path)
    file_ls = open(os.path.join('./', 'train_ls.txt' if 'train' in file_path.name else 'dev_ls.txt'), 'w')
    file_caps = open(os.path.join('./', 'train_caps.txt' if 'train' in file_path.name else 'dev_caps.txt'), 'w')
    for imgId, imgInfo in tqdm(coco_caps.imgs.items()):   # 397133, {'license': 4, 'file_name': '000000397133.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 'height': 427, 'width': 640, 'date_captured': '2013-11-14 17:02:52', 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'id': 397133}
        img_filename = coco_caps.loadImgs(ids=imgId)[0]['file_name']    # img文件名.jpg
        img_annIds = coco_caps.getAnnIds(imgIds=[imgId])    # img文本标注的annIds
        img_anns = coco_caps.loadAnns(ids=img_annIds)   # img文本标注数据结构
        if len(img_anns) > 5:
            img_anns = img_anns[:5]
        elif len(img_anns) < 5:
            print(f'Less than 5 captions! {file_path}--{img_filename}')
            img_anns += ['-------COPY-----' + ann_sturct for ann_sturct in img_anns[:5 - len(img_anns)]]
        for ann in img_anns:
            if keyword in ann['caption']:
                img_path = f'E:/迅雷下载/{"train2017" if "train" in file_path.name else "val2017"}/' + img_filename
                save_path = f"{os.path.dirname(img_path).replace('2017', '_' + keyword)}"
                os.makedirs(save_path, exist_ok=True)
                shutil.copy(img_path, os.path.join(save_path, img_filename))

                file_ls.write(img_path + '\n')
                [file_caps.write(ann['caption'].replace('\n', '') + '\n') for ann in img_anns]      # 某1行标注有许多空行
                # for ann in img_anns:
                #     if len(ann['caption']):
                #         file_caps.write('---------- Line ---------' + ann['caption'].rstrip('\n') + '\n')
                #     else:
                #         print(f'0-length string!!!!---->{file_path}--{img_filename}')
                #         exit(-1)
                break
    file_ls.close()
    file_caps.close()
    print(f'已生成:\n{file_ls.name}\n{file_caps.name}\n相关图片已拷贝到{save_path}\n共找到{len(os.listdir(save_path))}张图片.')
        # break
    # exit(-1)


if __name__ == '__main__':
    """
    自己需要修改的参数：
        keyword：文本标注的关键字，将带有此关键字的图片挑选出来
    """
    keyword = 'bicycle'
    sets = ['val2017', 'train2017']
    for set in sets:
        file_path = Path(f'./data/captions_{set}.json')
        get_by_class(file_path, keyword)
