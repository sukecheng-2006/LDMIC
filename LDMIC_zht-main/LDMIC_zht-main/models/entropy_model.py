import torch
import math
from torch import nn
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel, get_scale_table
from compressai.ops import quantize_ste
from compressai.layers import ResidualBlock, GDN, MaskedConv2d, conv3x3, ResidualBlockWithStride
import torch.nn.functional as F
import copy

class CheckMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask: A
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, mask_type: str = "A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')
        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == "A":
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        else:
            self.mask[:, :, 0::2, 0::2] = 1
            self.mask[:, :, 1::2, 1::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)
        return out

class Hyperprior(CompressionModel):
    
    def __init__(self, in_planes: int = 192, mid_planes: int = 192, out_planes: int=192):
        """_summary_
        整体介绍 __init__ 函数的功能
        __init__ 函数是 Python 中类的构造函数，用于初始化类的实例。
        在 Hyperprior 类中，__init__ 函数的主要功能是初始化超先验模型（Hyperprior Model）的网络结构。
        具体来说，它定义了超先验编码器（hyper_encoder）和超先验解码器（hyper_decoder）的结构，并调用了父类 CompressionModel 的构造函数来初始化熵瓶颈层（entropy_bottleneck）。
        #! Hpyerprior 从父类CompressionModel 中继承了一个属性 self.entropy_bottleneck = EntropyBottleneck, 是一个熵瓶颈网络层实例
        分析 __init__ 函数的参数
        __init__ 函数有三个参数：
        - in_planes (int, 默认值: 192): 输入特征图的通道数。
        - mid_planes (int, 默认值: 192): 中间特征图的通道数，通常用于编码器和解码器的中间层。
        - out_planes (int, 默认值: 192): 输出特征图的通道数。

        分析 __init__ 函数的返回值
        __init__ 函数没有显式的返回值。它的主要作用是初始化类的实例属性，特别是 hyper_encoder 和 hyper_decoder 这两个神经网络模块。

        高度概括 __init__ 函数的整体执行逻辑
        调用父类 CompressionModel 的构造函数，初始化熵瓶颈层。
        定义超先验编码器 hyper_encoder，它由多个卷积层和激活函数组成。
        根据 out_planes 的值，定义超先验解码器 hyper_decoder，它由多个反卷积层、激活函数和卷积层组成。
        """
        super().__init__(entropy_bottleneck_channels=mid_planes)    #* 这行代码调用了父类 CompressionModel 的构造函数，并传递了 entropy_bottleneck_channels=mid_planes 参数。mid_planes 是中间特征图的通道数，用于初始化熵瓶颈层
        #* 表明初始化了一个CompressionModel,他的超先验隐变量z的维度为通道数为mid_planes
        self.hyper_encoder = nn.Sequential(
            conv(in_planes, mid_planes, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
        )   #* 定义超先验编码器 hyper_encoder
        
        #* 下面是定义超先验解码器 hyper_decoder, 会根据out_planes的取值来定义hyper_decoder的结构
        if out_planes == 2 * in_planes:
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, in_planes * 3 // 2, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(in_planes * 3 // 2, out_planes, stride=1, kernel_size=3),
            )
        else:
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(mid_planes, out_planes, stride=1, kernel_size=3),
            )

    def forward(self, y, out_z=False):
        """_summary_
        ### 整体介绍 `forward` 函数的功能

        `forward` 函数是神经网络模型的核心方法，定义了数据在模型中的前向传播过程。
        在 `Hyperprior` 类中，`forward` 函数的主要功能是通过超先验编码器对输入数据进行编码，通过熵瓶颈层进行量化，并通过超先验解码器生成输出参数。
        此外，它还支持返回量化后的潜在表示（`z_hat`）和其对应的概率分布（`z_likelihoods`）。


        ### 分析 `forward` 函数的参数

        `forward` 函数有两个参数：
        1. **y (Tensor)**: 输入的特征图，通常是从主编码器输出的潜在表示。其形状为 `(batch_size, in_planes, H, W)`，其中：
        - `batch_size` 是批量大小。
        - `in_planes` 是输入通道数（默认为 192）。
        - `H` 和 `W` 分别是特征图的高度和宽度。
        2. **out_z (bool, 默认值: False)**: 一个布尔值，用于控制是否返回量化后的潜在表示 `z_hat`。如果为 `True`，则返回 `params, z_likelihoods, z_hat`；否则，只返回 `params, z_likelihoods`。


        ### 分析 `forward` 函数的返回值
        `forward` 函数的返回值取决于 `out_z` 参数的值：

        1. **如果 `out_z` 为 `False`**:
        - 返回一个元组 `(params, z_likelihoods)`，其中：
            - `params` 是超先验解码器生成的输出参数，形状为 `(batch_size, out_planes, H, W)`。
            - `z_likelihoods` 是量化后的潜在表示 `z_hat` 的概率分布，形状与 `z_hat` 相同。

        2. **如果 `out_z` 为 `True`**:
        - 返回一个元组 `(params, z_likelihoods, z_hat)`，其中：
            - `params` 和 `z_likelihoods` 同上。
            - `z_hat` 是量化后的潜在表示，形状为 `(batch_size, mid_planes, H/4, W/4)`。

        ### 高度概括 `forward` 函数的整体执行逻辑
        1. 通过超先验编码器 `hyper_encoder` 对输入 `y` 进行编码，得到潜在表示 `z`。
        2. 使用熵瓶颈层 `entropy_bottleneck` 对 `z` 进行量化，得到量化后的潜在表示 `z_hat` 和其概率分布 `z_likelihoods`。
        3. 通过超先验解码器 `hyper_decoder` 对 `z_hat` 进行解码，生成输出参数 `params`。
        4. 根据 `out_z` 参数的值，决定是否返回 `z_hat`。

        """
        z = self.hyper_encoder(y)   #* 输入隐变量y经过超先验编码器得到隐变量z
        #* 输入 y 的形状: (batch_size, in_planes, H, W), 输出 z 的形状: (batch_size, mid_planes, H/4, W/4)
        _, z_likelihoods = self.entropy_bottleneck(z)   #* 调用self.entropy_bottlenect(EntropyBottleneck类的对象)的forward函数，返回 量化后超先验 hat_z 和 量化后超先验的似然  z_likelihoods
        #* z_likelihoods 的形状与 z 相同，即 (batch_size, mid_planes, H/4, W/4)。
        z_offset = self.entropy_bottleneck._get_medians()   #* 得到分布的中位值
        z_hat = quantize_ste(z - z_offset) + z_offset   #* 使用 quantize_ste 函数对 z - z_offset 进行量化（STE 表示 Straight-Through Estimator，直通估计器）。将偏移量加回，得到量化后的潜在表示 z_hat。
        #* 量化后的超先验表示 z_hat 的维度为: (batch_size, mid_planes, H/4, W/4)
        params = self.hyper_decoder(z_hat)  #* 通过超先验解码器生成输出参数param, params 的形状: (batch_size, out_planes, H, W)
        #* 根据 out_z 参数决定返回值
        if out_z:   #* 如果 out_z 为 True:返回 params, z_likelihoods, z_hat
            return params, z_likelihoods, z_hat
        else:   #* #* 如果 out_z 为 False:返回 params, z_likelihoods,
            return params, z_likelihoods

    def compress(self, y):
        """_summary_
        ### 整体介绍 `compress` 函数的功能

        `compress` 函数用于将输入数据 `y` 压缩为二进制字符串（`z_strings`），并生成量化后的潜在表示 `z_hat` 和输出参数 `params`。
        该函数的主要作用是在编码阶段对输入数据进行压缩，以便于存储或传输。


        ### 分析 `compress` 函数的参数

        `compress` 函数有一个参数：

        1. **y (Tensor)**: 输入的特征图，通常是从主编码器输出的潜在表示。其形状为 `(batch_size, in_planes, H, W)`，其中：
        - `batch_size` 是批量大小。
        - `in_planes` 是输入通道数（默认为 192）。
        - `H` 和 `W` 分别是特征图的高度和宽度。


        ### 分析 `compress` 函数的返回值

        `compress` 函数返回一个元组 `(params, z_hat, z_strings)`，其中：

        1. **params (Tensor)**: 超先验解码器生成的输出参数，形状为 `(batch_size, out_planes, H, W)`。
        2. **z_hat (Tensor)**: 量化后的潜在表示，形状为 `(batch_size, mid_planes, H/4, W/4)`。
        3. **z_strings (list)**: 压缩后的二进制字符串列表，用于存储或传输。


        ### 高度概括 `compress` 函数的整体执行逻辑

        1. 通过超先验编码器 `hyper_encoder` 对输入 `y` 进行编码，得到潜在表示 `z`。
        2. 使用熵瓶颈层 `entropy_bottleneck` 对 `z` 进行压缩，生成二进制字符串 `z_strings`。
        3. 使用熵瓶颈层对 `z_strings` 进行解压缩，得到量化后的潜在表示 `z_hat`。
        4. 通过超先验解码器 `hyper_decoder` 对 `z_hat` 进行解码，生成输出参数 `params`。
        5. 返回 `params`、`z_hat` 和 `z_strings`。
        """
        z = self.hyper_encoder(y)   #* 通过超先验编码器 `hyper_encoder` 对输入 `y` 进行编码，得到超先验隐表示 `z`； 输出 z 的形状: (batch_size, mid_planes, H/4, W/4)
        z_strings = self.entropy_bottleneck.compress(z) #* 调用self.entropy_bottlenect(EntropyBottleneck类的对象)的compress函数对潜在表示 z 进行压缩，生成二进制字符串 z_strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])    #* 使用self.entropy_bottlenect(EntropyBottleneck类的对象)的decompress 方法对 z_strings 进行解压缩，得到量化后的潜在表示 z_hat
        params = self.hyper_decoder(z_hat)  #*  通过超先验解码器生成输出参数, 输出 params 的形状: (batch_size, out_planes, H, W)
        return params, z_hat, z_strings #{"strings": z_string, "shape": z.size()[-2:]}
        # 返回值:
        # params: 输出参数，形状为 (batch_size, out_planes, H, W)。
        # z_hat: 量化后的潜在表示，形状为 (batch_size, mid_planes, H/4, W/4)。
        # z_strings: 压缩后的二进制字符串列表。

    def decompress(self, strings, shape):
        #assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings, shape)
        params = self.hyper_decoder(z_hat)
        return params, z_hat
