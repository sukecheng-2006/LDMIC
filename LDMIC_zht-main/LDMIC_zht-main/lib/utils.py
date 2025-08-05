import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import torchvision.transforms.functional as tf
from typing import Dict
from torch import Tensor
import numpy as np
import glob
import cv2
import json
from PIL import Image
import random
import torch

class CropCityscapesArtefacts:
    """Crop Cityscapes images to remove artefacts"""

    def __init__(self):
        self.top = 64
        self.left = 128
        self.right = 128
        self.bottom = 256

    def __call__(self, image):
        """Crops a PIL image.
        Args:
            image (PIL.Image): Cityscapes image (or disparity map)
        Returns:
            PIL.Image: Cropped PIL Image
        """
        w, h = image.size
        assert w == 2048 and h == 1024, f'Expected (2048, 1024) image but got ({w}, {h}). Maybe the ordering of transforms is wrong?'
        #w, h = 1792, 704
        return image.crop((self.left, self.top, w-self.right, h-self.bottom))

class MinimalCrop:
    """
    Performs the minimal crop such that height and width are both divisible by min_div.
    """
    
    def __init__(self, min_div=16):
        self.min_div = min_div
        
    def __call__(self, image):
        w, h = image.size
        
        h_new = h - (h % self.min_div)
        w_new = w - (w % self.min_div)
        
        if h_new == 0 and w_new == 0:
            return image
        else:    
            h_diff = h-h_new
            w_diff = w-w_new

            top = int(h_diff/2)
            bottom = h_diff-top
            left = int(w_diff/2)
            right = w_diff-left

            return image.crop((left, top, w-right, h-bottom))

# class StereoImageDataset(Dataset):
#     """Dataset class for image compression datasets."""
#     #/home/xzhangga/datasets/Instereo2K/train/
#     def __init__(self, ds_type='train', ds_name='cityscapes', root='/home/xzhangga/datasets/Cityscapes/', crop_size=(256, 256), resize=False, **kwargs):
#         """
#         Args:
#             name (str): name of dataset, template: ds_name#ds_type. No '#' in ds_name or ds_type allowed. ds_type in (train, eval, test).
#             path (str): if given the dataset is loaded from path instead of by name.
#             transforms (Transform): transforms to apply to image
#             debug (bool, optional): If set to true, limits the list of files to 10. Defaults to False.
#         """
#         super().__init__()
        
#         self.path = Path(f"{root}")
#         self.ds_name = ds_name
#         self.ds_type = ds_type
#         print(ds_name)
#         if ds_type=="train":
#             self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
#                 transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
#         else: 
#             self.transform = transforms.Compose([transforms.ToTensor()])
#         self.left_image_list, self.right_image_list = self.get_files()


#         if ds_name == 'cityscapes':
#             self.crop = CropCityscapesArtefacts()
#         else:
#             if ds_type == "test":
#                 self.crop = MinimalCrop(min_div=64)
#             else:
#                 self.crop = None
#         #self.index_count = 0

#         print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.left_image_list)} files.')

#     def __len__(self):
#         return len(self.left_image_list)

#     def __getitem__(self, index):
#         #self.index_count += 1
#         image_list = [Image.open(self.left_image_list[index]).convert('RGB'), Image.open(self.right_image_list[index]).convert('RGB')]
#         if self.crop is not None:
#             image_list = [self.crop(image) for image in image_list]
#         frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
#         frames = torch.chunk(self.transform(frames), 2)
#         if random.random() < 0.5:
#             frames = frames[::-1]
#         return frames

#     def get_files(self):
#         if self.ds_name == 'cityscapes':
#             left_image_list, right_image_list, disparity_list = [], [], []
#             for left_image_path in self.path.glob(f'leftImg8bit/{self.ds_type}/*/*.png'):
#                 left_image_list.append(str(left_image_path))
#                 right_image_list.append(str(left_image_path).replace("leftImg8bit", 'rightImg8bit'))
#                 disparity_list.append(str(left_image_path).replace("leftImg8bit", 'disparity'))

#         elif self.ds_name == 'instereo2K':
#             path = self.path / self.ds_type
#             if self.ds_type == "test":
#                 folders = [f for f in path.iterdir() if f.is_dir()]
#             else:
#                 folders = [f for f in path.glob('*/*') if f.is_dir()]
#             left_image_list = [f / 'left.png' for f in folders]
#             right_image_list = [f / 'right.png' for f in folders]

#         elif self.ds_name == 'kitti':
#             left_image_list, right_image_list = [], []
#             ds_type = self.ds_type + "ing"
#             for left_image_path in self.path.glob(f'stereo2012/{ds_type}/colored_0/*.png'):
#                 left_image_list.append(str(left_image_path))
#                 right_image_list.append(str(left_image_path).replace("colored_0", 'colored_1'))

#             for left_image_path in self.path.glob(f'stereo2015/{ds_type}/image_2/*.png'):
#                 left_image_list.append(str(left_image_path))
#                 right_image_list.append(str(left_image_path).replace("image_2", 'image_3'))

#         elif self.ds_name == 'wildtrack':
#             C1_image_list, C4_image_list = [], []
#             for image_path in self.path.glob(f'images/C1/*.png'):
#                 if self.ds_type == "train" and int(image_path.stem) <= 2000:
#                     C1_image_list.append(str(image_path))
#                     C4_image_list.append(str(image_path).replace("C1", 'C4'))
#                 elif self.ds_type == "test" and int(image_path.stem) > 2000:
#                     C1_image_list.append(str(image_path))
#                     C4_image_list.append(str(image_path).replace("C1", 'C4'))
#             left_image_list, right_image_list = C1_image_list, C4_image_list
#         else:
#             raise NotImplementedError

#         return left_image_list, right_image_list
class StereoImageDataset(Dataset):
    """_summary_
    AdaptiveMultiCameraImageDataset 是一个自定义的 PyTorch Dataset 类，用于加载和处理多摄像头（multi-camera）图像数据集，支持训练和测试模式，
    并可适应不同数量的摄像头图像。该类可以用于训练深度学习模型，特别是与多视角图像相关的任务。
    """
    def __init__(self, ds_type='train',ds_name='wildtrack', root='/home/xzhangga/datasets/WildTrack/', crop_size=(256, 256),train_test_ratio = 4, **kwargs):
        """_summary_
        功能概述：
        该函数初始化了数据集对象，并根据不同的参数和数据集类型（训练或测试）设置相应的处理流程。
        
        参数：

        参数	说明
        ds_type	数据集类型，可选值为 'train' 或 'test'，决定数据加载方式。
        ds_name	数据集名称（默认为 wildtrack）。
        root	数据集根路径，指向数据存储目录。
        crop_size	随机裁剪大小（默认为 (256, 256)）。
        **kwargs	允许传递额外的参数。
        """
        super().__init__()
        
        self.path = Path(f"{root}") / ds_name#* 指定数据集的根目录。
        self.ds_type = ds_type  #* self.ds_type 表示 是“训练数据集” 还是 测试数据集
        if ds_type=="train":    #* ds_type == "train" 时，使用数据增强技术（随机裁剪、水平翻转、垂直翻转等）。
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
        else:   #* ds_type == "test" 时，只进行张量转换.
            self.transform = transforms.Compose([transforms.ToTensor()])
        #* 定义了self.transform， 用于对图像进行变换
        self.num_camera = 2   #*  调用 self.set_num_camera() 随机设置可用的摄像头数量。
        self.train_test_ratio = 1.0 * train_test_ratio / (train_test_ratio + 1)
        self.image_lists = self.get_files() #* self.image_lists 是一个  长度为 num_camera 的列表，其中每个元素都是一个列表，保存了某个摄像头拍摄的所有图像的路径。
        #* self.image_lists[i] 存放了 摄像头i拍摄的所有图像的路径
        #* 使用 self.get_files() 获取数据文件路径，并根据数据集类型设置不同的处理方式。
        if ds_type == "test":   
            #*  如果是测试集 (ds_type == "test")，使用 MinimalCrop 对图像进行裁剪。
            self.crop = MinimalCrop(min_div=64)
        else:
            self.crop = None

        print(f'Loaded dataset from {self.path}. Found {len(self.image_lists[0])} files.')
        print(f'Using {len(self.image_lists)} cameras.')
    def __len__(self):
        """_summary_
        功能： 
        返回数据集中的样本数量，即图像文件的数量。
        这里返回 self.image_lists[0] 的长度，因为它包含了第一个摄像头（C1）的图像路径，其他摄像头的路径数量应该相同。
        """
        return len(self.image_lists[0])

    def __getitem__(self, index):
        """_summary_
        整体介绍
        __getitem__ 函数是 Dataset 类的核心方法，用于从数据集中获取一个样本。
        其主要功能是根据数据集类型（训练或测试），加载指定数量的摄像头图像并进行一系列必要的图像处理（如裁剪、转换等）。
        最终，它返回一个包含处理后的图像的张量列表。

        分析函数参数
        index:
        该参数是从数据集中获取图像样本时的索引值。
        在 __getitem__ 中，index 用来访问 self.image_lists 中对应位置的图像路径。index 是一个整数值，表示要从 image_lists 中选取的图像的索引。
        
        
        返回值:

        frames: 这是一个包含多个图像张量的tuple，每个元素代表一个摄像头拍摄的图像。
        图像已经过必要的预处理操作（如裁剪、转换等）。每个图像被切割成多个部分（根据 num_camera），并将它们组合成一个tuple。
        #* frames 维度为: (self.num_camera,C, H,W); frames[i] 表示 第i个视点的第index个图像所对应的张量
        
        """
            #* 如果是训练集，使用 self.images_index（随机选取的摄像头索引）加载对应的图像
            #* self.images_index 是一个 长度为 self.num_camera 的列表， self.images_index[i] 的取值为[0~7]的一个随机数，且互相不重复
            #* self.image_lists[i] 表示 摄像头i拍摄的所有图像的路径，是一个列表； self.image_lists[i][index] 就是摄像头i拍摄的第index个图像
            #* 每个图像都使用 PIL.Image.open 打开，并通过 .convert('RGB') 转换为 RGB 格式。
            #* 如果是测试集，加载所有 num_camera 个摄像头的图像。
        image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in range(self.num_camera)]
            
        #* 至此，    image_list 是一个列表， 表示各个视点的第index张图像; image_list[i] 表示 第i个视点的第index张图像
            
        if self.crop is not None:
            image_list = [self.crop(image) for image in image_list] #* 如果设置了 self.crop（即裁剪函数），对每个加载的图像进行裁剪。

        frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
        # 将所有加载并处理过的图像（每个图像是一个 PIL.Image 对象）转换为 NumPy 数组，并在最后一个维度（axis=-1）拼接起来。
        #* np.asarray(image) 得到的结果为(H,W,3)
        # 例如，如果每个图像的大小是 (H, W, 3)，且有 3 个摄像头，拼接后 frames 的形状将变为 (H, W, 9)（假设每个图像有 3 通道，拼接后会有 9 个通道）。        
        frames = torch.chunk(self.transform(frames), self.num_camera)
        # 对拼接后的图像（frames）进行转换（如标准化、变换为张量等），转换后的张量将传递到 self.transform。self.transform 是一个由多种图像转换组成的变换函数（例如，将图像转换为 Tensor，进行标准化等）
        #* 由于self.transform 中 有Totensor, 因此: self.transform(frames) 结果为:  (9， H, W)（假设每个图像有 3 通道，拼接后会有 9 个通道）。 i.e. 通道在前
        #* ToTensor()：将图像从 NumPy 数组（或者 PIL 图像）转换为 PyTorch 张量，且将像素值从 [0, 255] 缩放到 [0, 1]
        # 输入格式： ToTensor() 接受两种常见的图像数据格式作为输入：
        # PIL 图像（Python Imaging Library 图像对象）。
        # NumPy 数组（形状通常为 (height, width, channels)）。
        # 输出格式： 转换后的输出是一个 PyTorch 张量（torch.Tensor），形状通常为 (channels, height, width)。即，它会将图像数据的通道维度（RGB）置于最前面，符合 PyTorch 中图像张量的标准格式。
        
        #* 然后，使用 torch.chunk 将图像张量按照 num_camera（摄像头的数量）切分成多个张量，每个张量代表一个摄像头的图像。

        #* frames 维度为: (self.num_camera,C, H,W); frames[i] 表示 第i个视点的第index个图像所对应的张量
        #if random.random() < 0.5:
        #    frames = frames[::-1]
        return frames

    def get_files(self):
        """_summary_
        整体介绍
        get_files 函数用于根据数据集名称和数据集类型（训练或测试），从指定的路径加载图像文件路径。它从 WildTrack 数据集中的多个摄像头视角获取图像路径，并按摄像头的不同存储图像路径到一个列表中，最后返回这些路径列表。

        分析函数参数
        num_camera (默认值 7):

        该参数指定数据集中使用的摄像头数量。通常为 7，代表 WildTrack 数据集中有 7 个不同的摄像头视角。它用于初始化一个包含多个摄像头图像路径的列表 image_lists，每个摄像头的路径都将存储在列表中的不同子列表中。
        分析函数返回值
        
        返回值:

        image_lists: 该函数返回一个长度为 num_camera 的列表，其中每个元素都是一个列表，保存了某个摄像头拍摄的所有图像的路径。image_lists[i] 存放了 摄像头i拍摄的所有图像的路径
        每个摄像头（即 C1, C2, ..., C7）都会有自己对应的图像路径列表。假设有 7 个摄像头，image_lists 的长度为 7，每个子列表分别包含该摄像头的图像路径。
        """
        image_lists = [[] for i in range(self.num_camera)]   #* 这行代码创建了一个包含 num_camera 个空列表的列表，num_camera 默认为 7。每个空列表用于存储一个摄像头的图像路径。
        for idx in range(1, self.num_camera + 1):
            path = self.path / f"C{idx}" 
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for image_path in path.glob(ext):  # 遍历当前路径下所有符合扩展名的图片
                    image_lists[idx - 1].append(str(image_path))  # 将图片路径添加到对应摄像头的列表中
        for idx in range(self.num_camera):
            train_num = (int) (self.train_test_ratio * len(image_lists[idx]))
            if(train_num % 2):
                train_num = train_num + 1
            if(self.ds_type == 'train'):
               image_lists[idx] = image_lists[idx][0:train_num]
            else:
                image_lists[idx] = image_lists[idx][train_num:]

        return image_lists #* image_lists 将被返回，它是一个长度为 num_camera（默认为 7）的列表，每个元素是一个子列表，包含该摄像头拍摄的图像路径


def save_checkpoint(state,is_best=False, log_dir=None, filename="ckpt.pth.tar"):
    save_file = os.path.join(log_dir, filename)
    print("save model in:", save_file)
    torch.save(state, save_file)
    if is_best:
        torch.save(state, os.path.join(log_dir, filename.replace(".pth.tar", ".best.pth.tar")))


# class MultiCameraImageDataset(Dataset):
#     def __init__(self, ds_type='train', ds_name='wildtrack', root='/home/xzhangga/datasets/WildTrack/', crop_size=(256, 256), num_camera=7, **kwargs):
#         super().__init__()
        
#         self.path = Path(f"{root}")
#         self.ds_name = ds_name
#         self.ds_type = ds_type
#         if ds_type=="train":
#             self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
#                 transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
#         else: 
#             self.transform = transforms.Compose([transforms.ToTensor()])
#         self.num_camera = num_camera
#         self.image_lists = self.get_files()
#         if ds_type == "test":
#             self.crop = MinimalCrop(min_div=64)
#         else:
#             self.crop = None

#         print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.image_lists[0])} files.')

#     def __len__(self):
#         return len(self.image_lists[0])

#     def __getitem__(self, index):
#         image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in range(self.num_camera)]
#         if self.crop is not None:
#             image_list = [self.crop(image) for image in image_list]
#         frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
#         frames = torch.chunk(self.transform(frames), self.num_camera)
#         #if random.random() < 0.5:
#         #    frames = frames[::-1]
#         return frames

#     def set_stage(self, stage):
#         if stage == 0:
#             print('Using (32, 32) crops')
#             self.crop = transforms.RandomCrop((32, 32))
#         elif stage == 1:
#             print('Using (28, 28) crops')
#             self.crop = transforms.RandomCrop((28, 28))

#     def get_files(self):
#         if self.ds_name == 'wildtrack':
#             image_lists = [[] for i in range(self.num_camera)]
#             for image_path in self.path.glob(f'images/C1/*.png'):
#                 if self.ds_type == "train" and int(image_path.stem) <= 2000:
#                     image_lists[0].append(str(image_path))
#                     for idx in range(1, self.num_camera):
#                         image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1))) 
#                 elif self.ds_type == "test" and int(image_path.stem) > 2000:
#                     image_lists[0].append(str(image_path))
#                     for idx in range(1, self.num_camera):
#                         image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1)))
#         else:
#             raise NotImplementedError

#         return image_lists


#     """_summary_
#     AdaptiveMultiCameraImageDataset 是一个自定义的 PyTorch Dataset 类，用于加载和处理多摄像头（multi-camera）图像数据集，支持训练和测试模式，
#     并可适应不同数量的摄像头图像。该类可以用于训练深度学习模型，特别是与多视角图像相关的任务。
#     """
#     def __init__(self, ds_type='train', ds_name='wildtrack', root='/home/xzhangga/datasets/WildTrack/', crop_size=(256, 256), num_camera=2, **kwargs):
#         """_summary_
#         功能概述：
#         该函数初始化了数据集对象，并根据不同的参数和数据集类型（训练或测试）设置相应的处理流程。
        
#         参数：

#         参数	说明
#         ds_type	数据集类型，可选值为 'train' 或 'test'，决定数据加载方式。
#         ds_name	数据集名称（默认为 wildtrack）。
#         root	数据集根路径，指向数据存储目录。
#         crop_size	随机裁剪大小（默认为 (256, 256)）。
#         **kwargs	允许传递额外的参数。
#         """
#         super().__init__()
        
#         self.path = Path(f"{root}") #* 指定数据集的根目录。
#         self.ds_name = ds_name  #* self.ds_name 表示 数据集的名字
#         self.ds_type = ds_type  #* self.ds_type 表示 是“训练数据集” 还是 测试数据集
#         if ds_type=="train":    #* ds_type == "train" 时，使用数据增强技术（随机裁剪、水平翻转、垂直翻转等）。
#             self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
#                 transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
#         else:   #* ds_type == "test" 时，只进行张量转换.
#             self.transform = transforms.Compose([transforms.ToTensor()])
#         #* 定义了self.transform， 用于对图像进行变换
#         self.num_camera = num_camera  #* self.num_camera 表示： 使用的摄像机的数目
#         self.set_num_camera()   #*  调用 self.set_num_camera() 随机设置可用的摄像头数量。        
#         self.image_lists = self.get_files() #* self.image_lists 是一个  长度为 num_camera 的列表，其中每个元素都是一个列表，保存了某个摄像头拍摄的所有图像的路径。
#         #* self.image_lists[i] 存放了 摄像头i拍摄的所有图像的路径
#         #* 使用 self.get_files() 获取数据文件路径，并根据数据集类型设置不同的处理方式。

#         if ds_type == "test":   
#             #*  如果是测试集 (ds_type == "test")，使用 MinimalCrop 对图像进行裁剪。
#             self.crop = MinimalCrop(min_div=64)
#             self.num_camera = 7
#         else:
#             self.crop = None

#         print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.image_lists[0])} files.')

#     def __len__(self):
#         """_summary_
#         功能： 
#         返回数据集中的样本数量，即图像文件的数量。
#         这里返回 self.image_lists[0] 的长度，因为它包含了第一个摄像头（C1）的图像路径，其他摄像头的路径数量应该相同。
#         """
#         return len(self.image_lists[0])

#     def __getitem__(self, index):
#         """_summary_
#         整体介绍
#         __getitem__ 函数是 Dataset 类的核心方法，用于从数据集中获取一个样本。
#         其主要功能是根据数据集类型（训练或测试），加载指定数量的摄像头图像并进行一系列必要的图像处理（如裁剪、转换等）。
#         最终，它返回一个包含处理后的图像的张量列表。

#         分析函数参数
#         index:
#         该参数是从数据集中获取图像样本时的索引值。
#         在 __getitem__ 中，index 用来访问 self.image_lists 中对应位置的图像路径。index 是一个整数值，表示要从 image_lists 中选取的图像的索引。
        
        
#         返回值:

#         frames: 这是一个包含多个图像张量的tuple，每个元素代表一个摄像头拍摄的图像。
#         图像已经过必要的预处理操作（如裁剪、转换等）。每个图像被切割成多个部分（根据 num_camera），并将它们组合成一个tuple。
#         #* frames 维度为: (self.num_camera,C, H,W); frames[i] 表示 第i个视点的第index个图像所对应的张量
        
#         """
#         if self.ds_type == "train":
#             #* 如果是训练集，使用 self.images_index（随机选取的摄像头索引）加载对应的图像
#             #* self.images_index 是一个 长度为 self.num_camera 的列表， self.images_index[i] 的取值为[0~7]的一个随机数，且互相不重复
#             #* self.image_lists[i] 表示 摄像头i拍摄的所有图像的路径，是一个列表； self.image_lists[i][index] 就是摄像头i拍摄的第index个图像
#             #* 每个图像都使用 PIL.Image.open 打开，并通过 .convert('RGB') 转换为 RGB 格式。
#             image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in self.images_index]
#         else:
#             #* 如果是测试集，加载所有 num_camera 个摄像头的图像。
#             image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in range(self.num_camera)]
            
#         #* 至此，    image_list 是一个列表， 表示各个视点的第index张图像; image_list[i] 表示 第i个视点的第index张图像
            
#         if self.crop is not None:
#             image_list = [self.crop(image) for image in image_list] #* 如果设置了 self.crop（即裁剪函数），对每个加载的图像进行裁剪。

#         frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
#         # 将所有加载并处理过的图像（每个图像是一个 PIL.Image 对象）转换为 NumPy 数组，并在最后一个维度（axis=-1）拼接起来。
#         #* np.asarray(image) 得到的结果为(H,W,3)
#         # 例如，如果每个图像的大小是 (H, W, 3)，且有 3 个摄像头，拼接后 frames 的形状将变为 (H, W, 9)（假设每个图像有 3 通道，拼接后会有 9 个通道）。        
#         frames = torch.chunk(self.transform(frames), self.num_camera)
#         # 对拼接后的图像（frames）进行转换（如标准化、变换为张量等），转换后的张量将传递到 self.transform。self.transform 是一个由多种图像转换组成的变换函数（例如，将图像转换为 Tensor，进行标准化等）
#         #* 由于self.transform 中 有Totensor, 因此: self.transform(frames) 结果为:  (9， H, W)（假设每个图像有 3 通道，拼接后会有 9 个通道）。 i.e. 通道在前
#         #* ToTensor()：将图像从 NumPy 数组（或者 PIL 图像）转换为 PyTorch 张量，且将像素值从 [0, 255] 缩放到 [0, 1]
#         # 输入格式： ToTensor() 接受两种常见的图像数据格式作为输入：
#         # PIL 图像（Python Imaging Library 图像对象）。
#         # NumPy 数组（形状通常为 (height, width, channels)）。
#         # 输出格式： 转换后的输出是一个 PyTorch 张量（torch.Tensor），形状通常为 (channels, height, width)。即，它会将图像数据的通道维度（RGB）置于最前面，符合 PyTorch 中图像张量的标准格式。
        
#         #* 然后，使用 torch.chunk 将图像张量按照 num_camera（摄像头的数量）切分成多个张量，每个张量代表一个摄像头的图像。

#         #* frames 维度为: (self.num_camera,C, H,W); frames[i] 表示 第i个视点的第index个图像所对应的张量
#         #if random.random() < 0.5:
#         #    frames = frames[::-1]
#         return frames

#     def set_num_camera(self):
#         """_summary_
#         功能：
#         随机设置可用摄像头的数量（num_camera），范围为 2 到 7 个摄像头。
#         随机选择 num_camera 个摄像头的索引，存储在 self.images_index 中。
        
#         #* self.num_camera 表示： 使用的摄像机的数目
#          #* self.images_index 是一个长度为self.num_camera 的不重复列表，self.images_index[i] 的取值为[0~7]的一个随机数，且互相不重复
#         """
        
#         #* random.randint(2, 7) 生成一个随机整数，表示使用的摄像头数量。
#         self.images_index = random.sample(range(7), self.num_camera)    #* self.images_index 是一个长度为self.num_camera 的不重复列表，self.images_index[i] 的取值为[0~7]的一个随机数
#         #* 从 7 个摄像头中随机选择 self.num_camera 个摄像头，并将这些摄像头的索引保存在 self.images_index 中。
#         #* random.sample(range(7), self.num_camera) 从这个[0~7] 7 个元素的列表中随机选择 self.num_camera 个唯一的索引，并返回一个包含这些索引的列表。
#         #* 注意：random.sample 保证选择的索引是唯一的，并且顺序是随机的，不会有重复。
#         #print("num_camera:",self.num_camera)

#     def get_files(self):
#         """_summary_
#         整体介绍
#         get_files 函数用于根据数据集名称和数据集类型（训练或测试），从指定的路径加载图像文件路径。它从 WildTrack 数据集中的多个摄像头视角获取图像路径，并按摄像头的不同存储图像路径到一个列表中，最后返回这些路径列表。

#         分析函数参数
#         num_camera (默认值 7):

#         该参数指定数据集中使用的摄像头数量。通常为 7，代表 WildTrack 数据集中有 7 个不同的摄像头视角。它用于初始化一个包含多个摄像头图像路径的列表 image_lists，每个摄像头的路径都将存储在列表中的不同子列表中。
#         分析函数返回值
        
#         返回值:

#         image_lists: 该函数返回一个长度为 num_camera 的列表，其中每个元素都是一个列表，保存了某个摄像头拍摄的所有图像的路径。image_lists[i] 存放了 摄像头i拍摄的所有图像的路径
#         每个摄像头（即 C1, C2, ..., C7）都会有自己对应的图像路径列表。假设有 7 个摄像头，image_lists 的长度为 7，每个子列表分别包含该摄像头的图像路径。
#         """
#         if self.ds_name == 'wildtrack':
#             image_lists = [[] for i in range(self.num_camera)]   #* 这行代码创建了一个包含 num_camera 个空列表的列表，num_camera 默认为 7。每个空列表用于存储一个摄像头的图像路径。
#             for image_path in self.path.glob(f'images/C1/*.png'):
#                 #* 使用 Path.glob() 方法从 self.path 指定的路径中查找符合条件的 .png 文件。
#                 #*这里的 self.path 是数据集的根路径，而 glob(f'images/C1/*.png') 会找到 images/C1/ 目录下的所有 .png 文件，这代表了第一摄像头（C1）拍摄的图像。
#                 #if self.ds_type == "train" and int(image_path.stem) <= 2000:    
#                 if self.ds_type == "train" and int(image_path.stem) < 200:     
#                     #* 图像的文件名编号小于或等于 2000 会被加入到训练集中
#                     image_lists[0].append(str(image_path))  #* 将 C1 摄像头拍摄的图像路径添加到 image_lists[0] 中。
#                     for idx in range(1, self.num_camera):
#                         #* 通过 for 循环将图像路径替换掉 C1 为其他摄像头（C2, C3, ... C7）的路径，并将这些路径添加到相应的 image_lists[idx] 中。
#                         image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1))) 
#                         # 例如，如果 image_path 为 images/C1/0001.png，则：
#                         # 对于 C1，image_lists[0] 中将添加 images/C1/0001.png
#                         # 对于 C2，image_lists[1] 中将添加 images/C2/0001.png
#                         # 对于 C3，image_lists[2] 中将添加 images/C3/0001.png
#                         # 依此类推。
#                 # elif self.ds_type == "test" and int(image_path.stem) > 2000:
#                 elif self.ds_type == "test" and int(image_path.stem) >= 200 and int(image_path.stem) <300:    
#                     #* 图像的文件名编号大于 2000 会被加入到测试集中
#                     image_lists[0].append(str(image_path))
#                     for idx in range(1, self.num_camera):
#                         image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1)))
#         else:
#             raise NotImplementedError

#         return image_lists #* image_lists 将被返回，它是一个长度为 num_camera（默认为 7）的列表，每个元素是一个子列表，包含该摄像头拍摄的图像路径。

# class AdaptiveMultiCameraImageDataset(Dataset):
#     """_summary_
#     AdaptiveMultiCameraImageDataset 是一个自定义的 PyTorch Dataset 类，用于加载和处理多摄像头（multi-camera）图像数据集，支持训练和测试模式，
#     并可适应不同数量的摄像头图像。该类可以用于训练深度学习模型，特别是与多视角图像相关的任务。
#     """
#     def __init__(self, ds_type='train', root='/home/xzhangga/datasets/WildTrack/', crop_size=(256, 256), num_camera=2, **kwargs):
#         """_summary_
#         功能概述：
#         该函数初始化了数据集对象，并根据不同的参数和数据集类型（训练或测试）设置相应的处理流程。
        
#         参数：

#         参数	说明
#         ds_type	数据集类型，可选值为 'train' 或 'test'，决定数据加载方式。
#         ds_name	数据集名称（默认为 wildtrack）。
#         root	数据集根路径，指向数据存储目录。
#         crop_size	随机裁剪大小（默认为 (256, 256)）。
#         **kwargs	允许传递额外的参数。
#         """
#         super().__init__()
        
#         self.path = Path(f"{root}") #* 指定数据集的根目录。
#         self.ds_type = ds_type  #* self.ds_type 表示 是“训练数据集” 还是 测试数据集
#         if ds_type=="train":    #* ds_type == "train" 时，使用数据增强技术（随机裁剪、水平翻转、垂直翻转等）。
#             self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
#                 transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
#         else:   #* ds_type == "test" 时，只进行张量转换.
#             self.transform = transforms.Compose([transforms.ToTensor()])
#         #* 定义了self.transform， 用于对图像进行变换
#         self.num_camera = num_camera   #*  调用 self.set_num_camera() 随机设置可用的摄像头数量。
#         self.check_num_camera() #* 检查数据集中的摄像头数量是否与 num_camera 参数一致。
#         self.image_lists = self.get_files() #* self.image_lists 是一个  长度为 num_camera 的列表，其中每个元素都是一个列表，保存了某个摄像头拍摄的所有图像的路径。
#         #* self.image_lists[i] 存放了 摄像头i拍摄的所有图像的路径
#         #* 使用 self.get_files() 获取数据文件路径，并根据数据集类型设置不同的处理方式。
#         if ds_type == "test":   
#             #*  如果是测试集 (ds_type == "test")，使用 MinimalCrop 对图像进行裁剪。
#             self.crop = MinimalCrop(min_div=64)
#         else:
#             self.crop = None

#         print(f'Loaded dataset from {self.path}. Found {len(self.image_lists[0])} files.')
#         print(f'Using {len(self.image_lists)} cameras.')
#     def check_num_camera(self):
#         count = 0
#         for item in os.listdir(self.path):
#             if item.startswith('C') and item[1:].isdigit():
#                 count += 1
#         if count < self.num_camera or self.num_camera <= 1:
#             print(f"Warning: {self.num_camera} cameras requested but only found {count} cameras in {self.path}")
#             raise NotImplementedError
#     def __len__(self):
#         """_summary_
#         功能： 
#         返回数据集中的样本数量，即图像文件的数量。
#         这里返回 self.image_lists[0] 的长度，因为它包含了第一个摄像头（C1）的图像路径，其他摄像头的路径数量应该相同。
#         """
#         return len(self.image_lists[0])

#     def __getitem__(self, index):
#         """_summary_
#         整体介绍
#         __getitem__ 函数是 Dataset 类的核心方法，用于从数据集中获取一个样本。
#         其主要功能是根据数据集类型（训练或测试），加载指定数量的摄像头图像并进行一系列必要的图像处理（如裁剪、转换等）。
#         最终，它返回一个包含处理后的图像的张量列表。

#         分析函数参数
#         index:
#         该参数是从数据集中获取图像样本时的索引值。
#         在 __getitem__ 中，index 用来访问 self.image_lists 中对应位置的图像路径。index 是一个整数值，表示要从 image_lists 中选取的图像的索引。
        
        
#         返回值:

#         frames: 这是一个包含多个图像张量的tuple，每个元素代表一个摄像头拍摄的图像。
#         图像已经过必要的预处理操作（如裁剪、转换等）。每个图像被切割成多个部分（根据 num_camera），并将它们组合成一个tuple。
#         #* frames 维度为: (self.num_camera,C, H,W); frames[i] 表示 第i个视点的第index个图像所对应的张量
        
#         """
#             #* 如果是训练集，使用 self.images_index（随机选取的摄像头索引）加载对应的图像
#             #* self.images_index 是一个 长度为 self.num_camera 的列表， self.images_index[i] 的取值为[0~7]的一个随机数，且互相不重复
#             #* self.image_lists[i] 表示 摄像头i拍摄的所有图像的路径，是一个列表； self.image_lists[i][index] 就是摄像头i拍摄的第index个图像
#             #* 每个图像都使用 PIL.Image.open 打开，并通过 .convert('RGB') 转换为 RGB 格式。
#             #* 如果是测试集，加载所有 num_camera 个摄像头的图像。
#         image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in range(self.num_camera)]
            
#         #* 至此，    image_list 是一个列表， 表示各个视点的第index张图像; image_list[i] 表示 第i个视点的第index张图像
            
#         if self.crop is not None:
#             image_list = [self.crop(image) for image in image_list] #* 如果设置了 self.crop（即裁剪函数），对每个加载的图像进行裁剪。

#         frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
#         # 将所有加载并处理过的图像（每个图像是一个 PIL.Image 对象）转换为 NumPy 数组，并在最后一个维度（axis=-1）拼接起来。
#         #* np.asarray(image) 得到的结果为(H,W,3)
#         # 例如，如果每个图像的大小是 (H, W, 3)，且有 3 个摄像头，拼接后 frames 的形状将变为 (H, W, 9)（假设每个图像有 3 通道，拼接后会有 9 个通道）。        
#         frames = torch.chunk(self.transform(frames), self.num_camera)
#         # 对拼接后的图像（frames）进行转换（如标准化、变换为张量等），转换后的张量将传递到 self.transform。self.transform 是一个由多种图像转换组成的变换函数（例如，将图像转换为 Tensor，进行标准化等）
#         #* 由于self.transform 中 有Totensor, 因此: self.transform(frames) 结果为:  (9， H, W)（假设每个图像有 3 通道，拼接后会有 9 个通道）。 i.e. 通道在前
#         #* ToTensor()：将图像从 NumPy 数组（或者 PIL 图像）转换为 PyTorch 张量，且将像素值从 [0, 255] 缩放到 [0, 1]
#         # 输入格式： ToTensor() 接受两种常见的图像数据格式作为输入：
#         # PIL 图像（Python Imaging Library 图像对象）。
#         # NumPy 数组（形状通常为 (height, width, channels)）。
#         # 输出格式： 转换后的输出是一个 PyTorch 张量（torch.Tensor），形状通常为 (channels, height, width)。即，它会将图像数据的通道维度（RGB）置于最前面，符合 PyTorch 中图像张量的标准格式。
        
#         #* 然后，使用 torch.chunk 将图像张量按照 num_camera（摄像头的数量）切分成多个张量，每个张量代表一个摄像头的图像。

#         #* frames 维度为: (self.num_camera,C, H,W); frames[i] 表示 第i个视点的第index个图像所对应的张量
#         #if random.random() < 0.5:
#         #    frames = frames[::-1]
#         return frames

#     def get_files(self):
#         """_summary_
#         整体介绍
#         get_files 函数用于根据数据集名称和数据集类型（训练或测试），从指定的路径加载图像文件路径。它从 WildTrack 数据集中的多个摄像头视角获取图像路径，并按摄像头的不同存储图像路径到一个列表中，最后返回这些路径列表。

#         分析函数参数
#         num_camera (默认值 7):

#         该参数指定数据集中使用的摄像头数量。通常为 7，代表 WildTrack 数据集中有 7 个不同的摄像头视角。它用于初始化一个包含多个摄像头图像路径的列表 image_lists，每个摄像头的路径都将存储在列表中的不同子列表中。
#         分析函数返回值
        
#         返回值:

#         image_lists: 该函数返回一个长度为 num_camera 的列表，其中每个元素都是一个列表，保存了某个摄像头拍摄的所有图像的路径。image_lists[i] 存放了 摄像头i拍摄的所有图像的路径
#         每个摄像头（即 C1, C2, ..., C7）都会有自己对应的图像路径列表。假设有 7 个摄像头，image_lists 的长度为 7，每个子列表分别包含该摄像头的图像路径。
#         """
#         num_camera = self.num_camera  #* 获取摄像头数量
#         image_lists = [[] for i in range(num_camera)]   #* 这行代码创建了一个包含 num_camera 个空列表的列表，num_camera 默认为 7。每个空列表用于存储一个摄像头的图像路径。
#         for idx in range(1, self.num_camera + 1):
#             path = self.path / f"C{idx}" 
#             train_folder = path / "train"
#             test_folder = path / "test"
#             for image_path in train_folder.glob("*.png"):
#                 if self.ds_type == "train":  
#                     image_lists[idx - 1].append(str(image_path))
#             for image_path in test_folder.glob("*.png"):
#                 if self.ds_type == "test":  
#                     image_lists[idx - 1].append(str(image_path))


#         return image_lists #* image_lists 将被返回，它是一个长度为 num_camera（默认为 7）的列表，每个元素是一个子列表，包含该摄像头拍摄的图像路径

# def save_checkpoint(state,is_best=False, log_dir=None, filename="ckpt.pth.tar"):
#     save_file = os.path.join(log_dir, filename)
#     print("save model in:", save_file)
#     torch.save(state, save_file)
#     if is_best:
#         torch.save(state, os.path.join(log_dir, filename.replace(".pth.tar", ".best.pth.tar")))

class MultiCameraImageDataset(Dataset):
    def __init__(self, ds_type='train', ds_name='wildtrack', root='D:/researchdata', crop_size=(256, 256), num_camera=7, dir_num = 2,
                 train_test_ratio = 4, **kwargs):
        super().__init__()
        
        self.path = Path(f"{root}") / ds_name
        self.ds_name = ds_name
        self.ds_type = ds_type
        self.train_test_ratio = 1.0 * train_test_ratio / (train_test_ratio + 1)
        if ds_type=="train":
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
        else: 
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.num_camera = num_camera
        self.dir_num = dir_num
        self.image_lists = self.get_files()

            
        if ds_type == "test":
            self.crop = MinimalCrop(min_div=64)
        else:
            self.crop = None

        print(f'Loaded {ds_type} dataset {ds_name} from {self.path}. Found {len(self.image_lists[0])} files.')

    def __len__(self):
        return len(self.image_lists[0])

    def __getitem__(self, index):
        image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in range(self.num_camera)]
        if self.crop is not None:
            image_list = [self.crop(image) for image in image_list]
        frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
        frames = torch.chunk(self.transform(frames), self.num_camera)
        #if random.random() < 0.5:
        #    frames = frames[::-1]
        return frames

    def set_stage(self, stage):
        if stage == 0:
            print('Using (32, 32) crops')
            self.crop = transforms.RandomCrop((32, 32))
        elif stage == 1:
            print('Using (28, 28) crops')
            self.crop = transforms.RandomCrop((28, 28))

    def get_files(self):
        dataset_lists = [[] for i in range(self.num_camera)]   #* 这行代码创建了一个包含 num_camera 个空列表的列表，num_camera 默认为 7。每个空列表用于存储一个摄像头的图像路径。
        image_lists = [[] for i in range(self.dir_num)]
        for idx in range(1, self.dir_num + 1):
            path = self.path / f"C{idx}" 
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for image_path in path.glob(ext):  # 遍历当前路径下所有符合扩展名的图片
                    image_lists[idx - 1].append(str(image_path))  # 将图片路径添加到对应摄像头的列表中
            
            cur_len = len(image_lists[idx-1])
            for ii in range(cur_len-self.num_camera+1):
                #* [ii, ii +num_camera-1]
                for jj in range(self.num_camera):
                    dataset_lists[jj].append(image_lists[idx-1][ii+jj])

        for idx in range(self.num_camera):
            train_num = (int) (self.train_test_ratio * len(dataset_lists[idx]))
            if(train_num % 2):
                train_num = train_num + 1
            if(self.ds_type == 'train'):
                dataset_lists[idx] = dataset_lists[idx][0:train_num]
            else:
                dataset_lists[idx] = dataset_lists[idx][train_num:]

        return dataset_lists #* image_lists 将被返回，它是一个长度为 num_camera（默认为 7）的列表，每个元素是一个子列表，包含该摄像头拍摄的图像路径

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_output_folder(parent_dir, env_name, output_current_folder=False):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    if not output_current_folder: 
        experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir, experiment_id

def save_tensors_to_file(tensors, filename):
    """
    将 PyTorch 张量变量写入文本文件。
    
    参数:
    tensors -- 一个字典，其中键是张量的名称，值是 PyTorch 张量。
    filename -- 要写入的文件名。
    """
    with open(filename, 'a') as f:
        for name, tensor in tensors.items():
            # 将张量转换为 NumPy 数组，然后写入文件
            f.write(f"{name}: {tensor.numpy()}\n")
            
def print_current_memory():
    allocated_memory = torch.cuda.memory_allocated()  # 当前已分配显存
    reserved_memory = torch.cuda.memory_reserved()    # 当前预留的显存
    # 打印最大分配的显存
    max_allocated_memory = torch.cuda.max_memory_allocated()    #* 返回程序运行期间的最大分配
    max_reserved_memory = torch.cuda.max_memory_reserved()  # 最大保留显存。这对于检查程序是否有显存泄漏或者训练过程中的内存需求非常有用。
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    total_memory_MB = total_memory / (1024 ** 2)

    print(f"GPU 总显存容量: {total_memory_MB:.2f} MB")
    print(f"当前已分配显存 (MB): {allocated_memory/1024/1024} MB")
    print(f"当前预留显存 (MB): {reserved_memory/1024/1024} MB")
    print(f"最大已分配显存 (MB): {max_allocated_memory/1024/1024} MB")
    print(f"最大预留显存 (MB): {max_reserved_memory/1024/1024} MB")