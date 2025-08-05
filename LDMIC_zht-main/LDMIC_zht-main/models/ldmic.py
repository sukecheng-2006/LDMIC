import torch
import math
#import __init__
from torch import nn
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel, get_scale_table
from compressai.ops import quantize_ste
from compressai.layers import ResidualBlock, GDN, MaskedConv2d, conv3x3, ResidualBlockWithStride
#from deepspeed.profiling.flops_profiler import get_model_profile
import torch.nn.functional as F
import sys
import os
from .entropy_model import Hyperprior, CheckMaskedConv2d
from torch.autograd import Variable
from math import exp
from compressai.ans import BufferedRansEncoder, RansDecoder
from lib.utils import save_tensors_to_file


# sys_path = os.getcwd()
# print(f"当前系统路径: {sys_path}")

# # 列出当前系统路径下的所有文件和文件夹
# files_and_folders = os.listdir(sys_path)

# 打印文件和文件夹列表
# for item in files_and_folders:
#     print(item)

class JointContextTransfer(nn.Module):
    """_summary_
    仅针对2个视图的JCT模块
    """
    def __init__(self, channels):
        super(JointContextTransfer, self).__init__()
        self.rb1 = ResidualBlock(channels, channels)    #* 第一个残差块
        self.rb2 = ResidualBlock(channels, channels)    #* 第二个残差块
        self.attn = EfficientAttention(key_in_channels=channels, query_in_channels=channels, key_channels=channels//8, 
            head_count=2, value_channels=channels//4)   #* 线性注意力模块

        self.refine = nn.Sequential(
            ResidualBlock(channels*2, channels),
            ResidualBlock(channels, channels))  #* 后处理网络

    def forward(self, x_left, x_right):
        """_summary_
        功能：
        联合上下文传递模块(JCT模块)的forward函数，
        
        参数：
        x_left: 左图对应的特征, 相当于文中的f
        x_right:右图对应的特征, 相当于文中的f
        
        返回值：
        JCT模块处理过后得到的特征:compact_left, compact_right, 相当于文中的f*
        
        """
        B, C, H, W = x_left.size()
        identity_left, identity_right = x_left, x_right
        x_left, x_right = self.rb2(self.rb1(x_left)), self.rb2(self.rb1(x_right))   #* x_left, x_right分别经过残差块self.rb1， self.rb2 的处理后，得到处理后特征x_left, x_right(相当于f')
        A_right_to_left, A_left_to_right = self.attn(x_left, x_right), self.attn(x_right, x_left)   #* 对x_left 和 x_right 做交叉注意力，得到
        compact_left = identity_left + self.refine(torch.cat((A_right_to_left, x_left), dim=1))
        compact_right = identity_right + self.refine(torch.cat((A_left_to_right, x_right), dim=1))
        return compact_left, compact_right


class Multi_JointContextTransfer(nn.Module):
    """_summary_
    `Multi_JointContextTransfer` 类是一个用于多视图的联合上下文传递模块（JCT模块）。
    它的主要功能是通过多视图之间的信息交互，增强每个视图的特征表示。
    具体来说，它通过残差块（Residual Block）提取特征，使用自适应平均池化和注意力机制（Efficient Attention）聚合多视图信息，并通过残差块进一步精炼特征。
    """
    def __init__(self, channels):
        super().__init__()
        self.rb = nn.Sequential(
            ResidualBlock(channels, channels),
            ResidualBlock(channels, channels),
        )
        self.aggeregate_module = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),
        )
        #* 这个层会对输入的3D张量在第一个维度上进行自适应平均池化，将其缩减为长度为1的维度。(1, None, None) 表示输出的第一个维度大小为1，而其他两个维度（高度和宽度）保持不变（None 表示保持原尺寸）。
        self.attn = EfficientAttention(key_in_channels=channels, query_in_channels=channels, key_channels=channels//8, 
            head_count=2, value_channels=channels//4)

        self.refine = nn.Sequential(
            ResidualBlock(channels*2, channels),
            ResidualBlock(channels, channels))

    def forward(self, x, num_camera):
        """_summary_
        功能：
        多视图的联合上下文传递模块(JCT模块)的forward函数，
        
        参数：
        1. **x (Tensor)**: 输入的特征张量，形状为 `(batch_size * num_camera, channels, height, width)`。其中：
        - `batch_size` 是批量大小。
        - `num_camera` 是视点数目。
        - `channels` 是特征通道数。
        - `height` 和 `width` 是特征图的高度和宽度。

        2. **num_camera (int)**: 视点数目，表示输入特征 `x` 中包含的视图数量。
        
        返回值：
        `forward` 函数返回一个张量，形状与输入 `x` 相同，即 `(batch_size * num_camera, channels, height, width)`。
        该张量是经过多视图联合上下文传递模块处理后的特征。
        
        ### 高度概括 `forward` 函数的整体执行逻辑

        1. 将输入特征 `x` 按视点数目 `num_camera` 切分为多个视图。
        2. 使用残差块提取每个视图的特征。
        3. 对于每个视图，聚合其他视图的特征，并通过注意力机制计算当前视图与其他视图的交互信息。
        4. 将交互信息与当前视图的特征结合，并通过残差块进一步精炼。
        5. 将所有视图的精炼特征拼接在一起，返回最终结果。
        
        """        
        identity_list = x.chunk(num_camera, 0)  #* 将输入特征 x 按视点数目 num_camera 切分为多个视图，存储在 identity_list 中。
        #* identity_list 是一个 长度为 num_camera 的列表，identity_list [i] 为一个维度为: (batch_size , channels, height, width) 的张量
        rb_x = self.rb(x)   #* 使用残差块 self.rb 提取特征，得到 rb_x
        rb_x_list = rb_x.chunk(num_camera, 0)   #* 将 rb_x 按视点数目 num_camera 切分为多个视图，存储在 rb_x_list 中
        #* identity_list 和 rb_x_list 是长度为 num_camera 的列表，每个元素的形状为 (batch_size, channels, height, width)
        compact_list = []   #* 用于存储每个视图的精炼特征。
        for idx, rb in enumerate(rb_x_list):    #* 遍历rb_x_list列表中的每一个精炼视图rb
            #* 对于每个视图 rb，执行以下操作：
            other_rb = [r.unsqueeze(2) for i, r in enumerate(rb_x_list) if i!=idx]
            other_rb = torch.cat(other_rb, dim=2)   #* 提取其他视图的特征，并将其拼接在一起，形成 other_rb。
            #* other_rb: 形状为 (batch_size, channels, num_camera - 1, height, width)
            aggeregate_rb = self.aggeregate_module(other_rb).squeeze(2) #* 使用自适应平均池化模块 self.aggeregate_module 聚合其他视图的特征，得到 aggeregate_rb
            #* aggeregate_rb: 形状为 (batch_size, channels, height, width)
            #print(rb.shape, aggeregate_rb.shape)
            A_other_camera_to_current = self.attn(rb, aggeregate_rb)    #* 使用交叉注意力机制 self.attn 计算当前视图 rb 与聚合特征 aggeregate_rb 的交互信息 A_other_camera_to_current
            #* A_other_camera_to_current: 形状为 (batch_size, channels, height, width)
            compact = identity_list[idx] + self.refine(torch.cat([A_other_camera_to_current, rb], dim=1))   
            #* 将交互信息 A_other_camera_to_current 与当前视图的特征 rb 结合，并通过残差块 self.refine 进一步精炼
            #* 将精炼后的特征与原始特征 identity_list[idx] 相加，得到最终的精炼特征 compact，并将其添加到 compact_list 中
            #* compact: 形状为 (batch_size, channels, height, width)
            compact_list.append(compact)
        
        return torch.cat(compact_list, dim=0)   #* 将所有视图的精炼特征 compact_list 拼接在一起，返回最终结果
        #* 返回的张量形状为 (batch_size * num_camera, channels, height, width)


class EfficientAttention(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, key_channels=32, head_count=8, value_channels=64):
        super().__init__()
        self.in_channels = query_in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(key_in_channels, key_channels, 1)
        self.queries = nn.Conv2d(query_in_channels, key_channels, 1)
        self.values = nn.Conv2d(key_in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, query_in_channels, 1)

    def forward(self, target, input):
        """_summary_
        整体介绍
        EfficientAttention 类中的 forward 函数实现了一个多头注意力机制的计算。它通过对目标（target）和输入（input）进行一系列卷积操作，计算得到注意力（attention）的输出。

        注意力机制旨在计算“加权的输入信息”，在这里我们通过对 input 和 target 进行一系列卷积变换后，分别计算 key、query 和 value，然后通过加权的方式得到最终的 attention 输出。

        函数参数
        target：形状为 (n, c, h, w)，表示目标输入（可能是图像的某一部分或特定特征图），这里 n 是批次大小，c 是通道数，h 和 w 是图像的高度和宽度。

        input：形状为 (n, c, h, w)，表示输入特征图，通常是图像或特征图的不同部分。

        返回值
        attention：形状为 (n, c, h, w)，表示计算得到的注意力图，通过卷积操作得到的加权输入特征。
        """
        n, _, h, w = input.size()
        # input.size() 返回四维张量的维度 (n, c, h, w)，其中：
        # n 是批次大小（batch size），
        # c 是通道数，
        # h 和 w 分别是高度和宽度。
        keys = self.keys(input).reshape((n, self.key_channels, h * w))  #* keys 的维度: (n, key_channels, h * w); 
        queries = self.queries(target).reshape(n, self.key_channels, h * w)  #* queries 的维度: (n, key_channels, h * w); 
        values = self.values(input).reshape((n, self.value_channels, h * w))    #* values 的维度: (n, value_channels, h * w); 
        #* self.keys(input)，self.queries(target) 和 self.values(input) 分别通过卷积得到形状为 (n, key_channels, h, w)、(n, key_channels, h, w) 和 (n, value_channels, h, w) 的张量。
        #* 然后，通过 reshape 将它们展平为形状为 (n, key_channels, h * w)、(n, key_channels, h * w) 和 (n, value_channels, h * w) 的张量。
        
        head_key_channels = self.key_channels // self.head_count    #* head_key_channels 表示每个注意力头的 key  的通道数。
        head_value_channels = self.value_channels // self.head_count    #* head_value_channels 表示每个注意力头的  value 的通道数。
        
        attended_values = []
        for i in range(self.head_count):
            #* 对每个头，分割 keys、queries 和 values，并对其进行 softmax 归一化。
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels,:], dim=2)   #* key的维度: (n, head_key_channels, h * w)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels,:], dim=1)  #* query的维度: (n, head_key_channels, h * w)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]    #* value的维度: (n, head_value_channels, h * w)
            context = key @ value.transpose(1, 2)   #* context = key @ value.transpose(1, 2)：计算 key 和 value 的点积，得到上下文信息。
            #* 对于批量矩阵乘法 A @ B，如果 A 的维度是 (n, a, m)，B 的维度是 (n, m, b)，则结果的维度是 (n, a, b)。
            #* context的维度为: (n, head_key_channels, head_value_channels)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            #* (context.transpose(1, 2) @ query)：将上下文与 query 点积，得到最终的注意力值, 维度为(n, head_value_channels, hw)
            #* attended_value 维度: (n, head_value_channels, h,w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)   #* 将所有头的输出在通道维度（dim=1）上拼接。
        #* aggregated_values 维度为: (n, self.value_channels, h,w)
        reprojected_value = self.reprojection(aggregated_values)    #* reprojected_value 维度: (n, query_in_channels, h,w)
        attention = reprojected_value #+ input_

        return attention

    def parallel_forward(self, target, input):
        n, _, h, w = input.size()
        keys = self.keys(input).reshape((n, self.key_channels, h * w))
        queries = self.queries(target).reshape(n, self.key_channels, h * w)
        values = self.values(input).reshape((n, self.value_channels, h * w))
        
        keys = keys.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)
        queries = queries.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)
        values = values.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)

        key = F.softmax(keys, dim=2)
        queries = F.softmax(queries, dim=1)
        context = key @ value.transpose(1, 2)
        attended_values = (context.transpose(1, 2) @ query).reshape(n, -1, h, w)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value #+ target
        return attention


class LDMIC(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=JointContextTransfer):
        """_summary_
        __init__ 函数是 LDMIC 类的初始化方法，用于定义和初始化该类的各个组件和参数。
        LDMIC 是一个基于深度学习的图像压缩模型，其主要功能是通过编码器、超先验模块、上下文预测模块、熵参数模块以及解码器等组件，实现图像的压缩和解压缩过程。
        
        执行逻辑：
        __init__ 函数的主要执行逻辑是初始化 LDMIC 类的各个组件，包括编码器、超先验模块、上下文预测模块、熵参数模块、高斯条件模块、注意力模块以及解码器。这些组件共同构成了图像压缩模型的架构。
        
        
        Args:
        N：整数，默认值为 128。表示编码器中间层的通道数。
        M：整数，默认值为 192。表示编码器输出层的通道数，同时也是超先验模块的输入通道数。
        decode_atten：类，默认值为 JointContextTransfer。表示用于解码过程中使用的注意力模块。
        """
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )
        # 输入图像的维度为 (batch_size, 3, height, width)。
        # 第一层卷积后，输出维度为 (batch_size, N, height/2, width/2)。
        # 第二层卷积后，输出维度为 (batch_size, N, height/4, width/4)。
        # 第三层卷积后，输出维度为 (batch_size, N, height/8, width/8)。
        # 第四层卷积后，输出维度为 (batch_size, M, height/16, width/16)。
        #* 举例：
        # 假设输入图像大小为 (1, 3, 256, 256)，N=128，M=192。
        # 经过编码器后，输出特征图的大小为 (1, 192, 16, 16)。
        
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2) #* 自定义的超先验模块
        # 变量维度分析：
        # 输入特征图的维度为 (batch_size, M, height/16, width/16)。
        # 输出的超先验特征图维度为 (batch_size, M*2, height/32, width/32)。
        # 举例：
        # 输入特征图大小为 (1, 192, 16, 16)。
        # 输出超先验特征图的大小为 (1, 384, 8, 8)。
        
        self.context_prediction = MaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )   #* 自定义的上下文预测模块
        # 变量维度分析：
        # 输入特征图的维度为 (batch_size, M, height/16, width/16)。
        # 输出特征图的维度为 (batch_size, M*2, height/16, width/16)。
        # 举例：
        # 输入特征图大小为 (1, 192, 16, 16)。
        # 输出特征图的大小为 (1, 384, 16, 16)。
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )   #* 自定义的熵参数模块
        # 变量维度分析：
        # 输入特征图的维度为 (batch_size, M*4, height/16, width/16)。
        # 经过第一层卷积后，输出维度为 (batch_size, M*10/3, height/16, width/16)。
        # 经过第二层卷积后，输出维度为 (batch_size, M*8/3, height/16, width/16)。
        # 经过第三层卷积后，输出维度为 (batch_size, M*6/3, height/16, width/16)。
        # 举例：
        # 输入特征图大小为 (1, 768, 16, 16)。
        # 输出特征图的大小为 (1, 384, 16, 16)。
        self.gaussian_conditional = GaussianConditional(None)   #* 条件高斯模块，用于使用高斯分布来做熵编码
        self.M = M
        self.atten_3 = decode_atten(M)  #* 这里的atten也即JCT(联合上下文传递模块)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)  #* 第一阶段解码器，用于计算当前视图的隐特征
        self.atten_4 = decode_atten(N)  #* 这里的atten也即JCT(联合上下文传递模块)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )   #* 第二阶段解码器，用于计算重建图像

    def forward(self, x):
        """_summary_
        整体功能介绍
        forward 函数是 LDMIC 类的核心方法，用于实现图像压缩模型的前向传播过程。
        它接收输入图像对（左图和右图），通过编码器、超先验模块、上下文预测模块、熵参数模块、高斯条件模块、注意力模块和解码器等组件，
        完成图像的压缩和重建，并计算相关概率分布的似然值。

        参数分析
        x：输入数据，是一个包含两个图像张量的列表，x[0] 表示左图，x[1] 表示右图。
        每个图像张量的维度为 (batch_size, 3, height, width), 表示一个大小为batch_size 的小批量， 通道数为3，高度为height, 宽度为width.

        返回值:
        返回一个字典，包含以下内容：
            x_hat：重建的图像对，x_hat[0] 是重建的左图，x_hat[1] 是重建的右图。
            likelihoods：包含左图和右图的似然值，每个图的似然值是一个字典，包含：
                y：编码后的特征图的似然值。
                z：超先验特征图的似然值。
            feature：用于调试或分析的中间特征，包括：
                y_left_ste 和 y_right_ste：经过符号梯度估计（STE）的特征图。
                z_left_hat 和 z_right_hat：超先验特征图的重建版本。
                left_means_hat 和 right_means_hat：高斯分布的均值。
                
        整体执行逻辑:
        将输入图像对(pair)分别通过编码器进行编码。
        使用超先验模块对编码后的特征图进行处理，得到超先验特征图及其似然值。
        对编码后的特征图进行量化，并通过上下文预测模块和熵参数模块计算高斯分布的参数。
        使用高斯条件模块计算编码特征图的似然值。
        对量化后的特征图进行符号梯度估计（STE），并通过注意力模块和解码器重建图像。
        返回重建图像、似然值和中间特征。                
        """
        x_left, x_right = x[0], x[1] #x.chunk(2, 1) #* 将输入图像对分离为左图和右图。
        # 变量维度：
        # x_left 和 x_right 的维度均为 (batch_size, 3, height, width)。

        y_left, y_right = self.encoder(x_left), self.encoder(x_right)   #* 将左图和右图分别通过编码器进行编码。y_left 和 y_right 的维度均为 (batch_size, M, height/16, width/16)。
        #
        left_params, z_left_likelihoods, z_left_hat = self.hyperprior(y_left, out_z=True)
        right_params, z_right_likelihoods, z_right_hat = self.hyperprior(y_right, out_z=True)
        #* 对编码后的特征图 y_left 和 y_right 进行超先验处理，得到(超先验特征图z)及其(似然值z_likelihoods) 以及经过超先验解码器hyperdecoder计算得到的(处理后超先验left_params/right_params)
        # 变量维度：
        # left_params 和 right_params 的维度均为 (batch_size, M*2, height/32, width/32)。
        # z_left_likelihoods 和 z_right_likelihoods 的维度均为 (batch_size, M*2, height/32, width/32)。
        # z_left_hat 和 z_right_hat 的维度均为 (batch_size, M*2, height/32, width/32)。
        
        #
        y_left_hat = self.gaussian_conditional.quantize(
            y_left, "noise" if self.training else "dequantize"
        )
        y_right_hat = self.gaussian_conditional.quantize(
            y_right, "noise" if self.training else "dequantize"
        )
        #* 对编码后的特征图进行量化操作。
        # 变量维度：
        # y_left_hat 和 y_right_hat 的维度均为 (batch_size, M, height/16, width/16)。
        
        ctx_left_params = self.context_prediction(y_left_hat)
        ctx_right_params = self.context_prediction(y_right_hat)
        #* 使用上下文预测模块生成上下文参数。
        # 变量维度：
        # ctx_left_params 和 ctx_right_params 的维度均为 (batch_size, M*2, height/16, width/16)。        
        
        

        gaussian_left_params = self.entropy_parameters(torch.cat([left_params, ctx_left_params], 1))
        gaussian_right_params = self.entropy_parameters(torch.cat([right_params, ctx_right_params], 1))
        #* 将超先验参数和上下文参数拼接后，通过熵参数模块计算高斯分布的参数。
        # 变量维度：
        # gaussian_left_params 和 gaussian_right_params 的维度均为 (batch_size, M*2, height/16, width/16)。        
        
        left_means_hat, left_scales_hat = gaussian_left_params.chunk(2, 1)
        right_means_hat, right_scales_hat = gaussian_right_params.chunk(2, 1)
        #* 将高斯分布参数分离为均值和尺度。
        # 变量维度：
        # left_means_hat 和 right_means_hat 的维度均为 (batch_size, M, height/16, width/16)。
        # left_scales_hat 和 right_scales_hat 的维度均为 (batch_size, M, height/16, width/16)。        
        
         
        _, y_left_likelihoods = self.gaussian_conditional(y_left, left_scales_hat, means=left_means_hat)
        _, y_right_likelihoods = self.gaussian_conditional(y_right, right_scales_hat, means=right_means_hat)
        #* 使用高斯条件模块计算编码特征图的似然值。
        # 变量维度：
        # y_left_likelihoods 和 y_right_likelihoods 的维度均为 (batch_size, M, height/16, width/16)。

        y_left_ste, y_right_ste = quantize_ste(y_left - left_means_hat) + left_means_hat, quantize_ste(y_right - right_means_hat) + right_means_hat
        #* 对编码特征图进行符号梯度估计（STE）。 quantize_ste(x)实现的就是对符号x的量化round操作，只不过它是带有梯度的量化，这个梯度是用恒等映射来近似的。
        # 变量维度：
        # y_left_ste 和 y_right_ste 的维度均为 (batch_size, M, height/16, width/16)。
        y_left, y_right = self.atten_3(y_left_ste, y_right_ste)
        
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        # 经过 decoder_1 后，y_left 和 y_right 的维度均为 (batch_size, N, height/4, width/4)
        #* 使用JCT模块对左右视图的特征进行融合，得到增强后的特征 y_left, y_right
        
        
        x_left_hat, x_right_hat = self.decoder_2(y_left), self.decoder_2(y_right)
        #* 使用第二阶段解码器，以增强后的特征作为输入，输出重建图像 x_left_hat, x_right_hat
        return {
            "x_hat": [x_left_hat, x_right_hat],
            "likelihoods": [{"y": y_left_likelihoods, "z": z_left_likelihoods}, {"y":y_right_likelihoods, "z":z_right_likelihoods}],
            "feature": [y_left_ste, y_right_ste, z_left_hat, z_right_hat, left_means_hat, right_means_hat],
        }   #* 返回重建图像、似然值和中间特征。
    """
    forward: 前向传播函数，实现图像压缩模型的前向传播过程。
    为什么对于y是直接进行量化？而非先padding再量化？
    forward注重于训练过程，而不是压缩过程。他的目的是计算包括重建图像，相关的似然值和中间特征，用于训练时的损失计算和模型参数更新。
    因此我们所追求的应该是去模拟整个压缩过程，最大程度的还原原图像
    同时在性能上，直接量化后的图像，可以减少计算量，提高训练速度。
    可能担心的问题是，直接量化后的图像，会丢失一些信息，但是这些信息对于训练来说，可能并不是那么重要。我们还有超先验过程为我们兜底，即使丢失了一些信息，也不会对训练产生太大的影响。
    相对于padding之后带来的计算量的增加，与他所带来的信息增加，我们更倾向于选择直接量化。如果进行padding之后的量化，我们后面也应会有对应的反量化操作这样会增加恐怖的计算量
    甚至在此代码中为了减少计算量，直接使用了STE来得到 y_left_ste，近乎省去了关于y的AE-AD操作
    总结来说：forward更侧重于模拟完整的压缩和重建流程，重点在于验证模型能否有效地高效的学习到图像的特征表示和重建能力，而不是追求极致的压缩效率。
    """
    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def fix_encoder(self):
        """
        整体功能介绍
        fix_encoder 函数的作用是将模型中编码器及其相关模块的参数设置为不可训练状态。这通常用于模型的微调阶段，当需要固定某些模块的参数，而只训练其他模块时，可以使用此函数。
        
        参数分析
        该函数没有输入参数。
        
        返回值
        该函数没有返回值，其作用是直接修改模型中各个模块的参数属性。
        
        整体执行逻辑
        fix_encoder 函数通过遍历模型中编码器及其相关模块的所有参数，并将它们的 requires_grad 属性设置为 False，从而固定这些参数，使其在后续的训练过程中不会更新。
        """
        for p in self.encoder.parameters(): #* 遍历 self.encoder 中的所有参数，并将它们的 requires_grad 属性设置为 False
            p.requires_grad = False 
        for p in self.hyperprior.parameters():
            p.requires_grad = False
        for p in self.context_prediction.parameters():
            p.requires_grad = False
        for p in self.entropy_parameters.parameters():
            p.requires_grad = False
        for p in self.gaussian_conditional.parameters():
            p.requires_grad = False

    def load_encoder(self, current_state_dict, checkpoint):
        """_summary_
        整体功能介绍
        load_encoder 函数的作用是从一个预训练的检查点（checkpoint）中加载与编码器及其相关模块相关的参数，并将这些参数更新到当前模型的状态字典（current_state_dict）中。
        这个函数主要用于模型的参数迁移，例如从一个预训练模型加载特定模块的参数。
        
        参数分析
        current_state_dict：当前模型的状态字典，包含模型的所有参数。类型为 dict。
        checkpoint：预训练模型的检查点，包含预训练模型的所有参数。类型为 dict。
        
        返回值
        返回更新后的当前模型状态字典 current_state_dict。
        
        整体执行逻辑
        从 checkpoint 中提取与编码器及其相关模块（超先验模块、上下文预测模块、熵参数模块、高斯条件模块）相关的参数。
        将提取的参数映射到当前模型的对应模块。
        使用update函数更新当前模型的状态字典。
        """
        encoder_dict = {k.replace("g_a", "encoder"): v for k, v in checkpoint.items() if "g_a" in k}
        #* 从 checkpoint 中提取键中包含 "g_a" 的参数，并将键名替换为 "encoder"
        #* 举例：假设 checkpoint 中有参数 "g_a.conv1.weight"，则 encoder_dict 中对应的键为 "encoder.conv1.weight"。

        context_prediction_dict = {k: v for k, v in checkpoint.items() if "context_prediction" in k}    #* 从 checkpoint 中提取键中包含 "context_prediction" 的参数。
        entropy_parameters_dict = {k: v for k, v in checkpoint.items() if "entropy_parameters" in k}    #* 从 checkpoint 中提取键中包含 "entropy_parameters" 的参数
        gaussian_conditional_dict = {k: v for k, v in checkpoint.items() if "gaussian_conditional" in k}
        
        hyperprior_dict = {}
        #* 下面一段代码从 checkpoint 中提取与超先验模块相关的参数，并将键名映射到当前模型的超先验模块
        for k, v in checkpoint.items():
            if "h_a" in k:
                hyperprior_dict[k.replace("h_a", "hyperprior.hyper_encoder")] = v
            elif "h_s" in k:
                hyperprior_dict[k.replace("h_s", "hyperprior.hyper_decoder")] = v
            elif "entropy_bottleneck" in k:
                hyperprior_dict[k.replace("entropy_bottleneck", "hyperprior.entropy_bottleneck")] = v
        #* 举例：
        # 假设 checkpoint 中有参数 "h_a.conv1.weight"，则 hyperprior_dict 中对应的键为 "hyperprior.hyper_encoder.conv1.weight"。
        # 假设 checkpoint 中有参数 "h_s.conv1.weight"，则 hyperprior_dict 中对应的键为 "hyperprior.hyper_decoder.conv1.weight"。
        # 假设 checkpoint 中有参数 "entropy_bottleneck.weight"，则 hyperprior_dict 中对应的键为 "hyperprior.entropy_bottleneck.weight"。

        #* 使用update方法将提取的参数字典逐一更新到当前模型的状态字典中。
        current_state_dict.update(encoder_dict)
        current_state_dict.update(hyperprior_dict)
        current_state_dict.update(context_prediction_dict)
        current_state_dict.update(entropy_parameters_dict)
        current_state_dict.update(gaussian_conditional_dict)
        #print(current_state_dict.keys())
        #input()
        return current_state_dict

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated

    def compress(self, x):
        x_left, x_right = x[0], x[1]
        left_dict = self.encode(x_left)
        right_dict = self.encode(x_right)
        return left_dict, right_dict

    def decompress(self, left_dict, right_dict):
        y_left_hat = self.decode(left_dict["strings"], left_dict["shape"])
        y_right_hat = self.decode(right_dict["strings"], right_dict["shape"])
        y_left, y_right = self.atten_3(y_left_hat, y_right_hat)
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        x_left_hat, x_right_hat = self.decoder_2(y_left).clamp_(0, 1), self.decoder_2(y_right).clamp_(0, 1)
        return {
            "x_hat": [x_left_hat, x_right_hat],
        }  

    def encode(self, x):
        """_summary_
        整体功能介绍
        encode 函数的作用是将输入图像 x 编码为一组字符串，这些字符串可以用于后续的解码和重建图像。编码过程包括以下步骤：
        - 使用编码器对输入图像进行编码。
        - 使用超先验模块对编码后的特征图进行压缩，得到超先验特征图及其压缩表示。
        - 对编码后的特征图进行上下文预测和自回归压缩。
        - 返回压缩后的字符串和特征图的形状信息。
        
        参数分析
        x：输入图像张量，维度为 (batch_size, 3, height, width)。
        
        返回值
        返回一个字典，包含以下内容：
        - strings：一个列表，包含两部分：
        -- y_strings：编码后的特征图的压缩字符串。
        -- z_strings：超先验特征图的压缩字符串。
        - shape：超先验特征图的形状信息，具体为 (height, width)。
        - 说明：shape 表示超先验特征图的空间维度，用于后续解码时恢复特征图的大小。
        
        整体执行逻辑
        使用编码器对输入图像进行编码。
        使用超先验模块对编码后的特征图进行压缩，得到超先验特征图及其压缩表示。
        对编码后的特征图进行上下文预测和自回归压缩。
        返回压缩后的字符串和特征图的形状信息。
        """
        y = self.encoder(x) #* 使用编码器对输入图像 x 进行编码，得到隐特征y
        # 变量维度：
        # 输入 x 的维度为 (batch_size, 3, height, width)。
        # 输出 y 的维度为 (batch_size, M, height/16, width/16)。
        params, z_hat, z_strings = self.hyperprior.compress(y)  
        #* 使用超先验模块对编码后的特征图 y 进行压缩，得到: 经过超先验解码器处理后的特征params(用于后续预测熵模型参数), 超先验特征图 z_hat 及其压缩表示 z_strings。
        # 变量维度：
        # 输入 y 的维度为 (batch_size, M, height/16, width/16)。
        # 输出 z_hat 的维度为 (batch_size, M*2, height/32, width/32)。
        # 输出 z_strings 是一个包含压缩字符串的列表。        
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))  #* 对隐特征y的上下左右进行填充，得到y_hat(填充后隐特征)
        #* 使用 PyTorch 的 F.pad 函数对编码特征图 y 进行填充，以便后续的上下文预测。
        # F.pad 是 PyTorch 中的一个函数，用于对张量进行填充（padding）。填充操作通常用于在张量的边界添加额外的值，以便在卷积操作中保持张量的尺寸不变，或者为某些操作提供额外的上下文信息。
        # torch.nn.functional.pad(input, pad, mode='constant', value=0) input：需要填充的张量。 pad：一个元组，指定填充的大小。对于二维张量（如图像），pad 通常是一个长度为 4 的元组 (left, right, top, bottom)，分别表示在左边、右边、顶部和底部填充的大小。
        # 变量维度：
        # 输入 y 的维度为 (batch_size, M, height/16, width/16)。
        # 输出 y_hat 的维度为 (batch_size, M, height/16 + 2*padding, width/16 + 2*padding)。
        # 举例：
        # 假设输入特征图大小为 (1, 192, 16, 16)，padding=2。
        # 输出特征图的大小为 (1, 192, 20, 20)。        

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],   #* 选取y_hat当前小批量中的一个样本y_hat[i]
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )   #*  _compress_ar 函数的作用是实现自回归（Auto-Regressive）压缩。它通过上下文预测模块和熵参数模块，对输入的特征图 y_hat 进行逐像素的压缩。 
            #* 该函数利用高斯条件模块计算每个像素的量化值，并通过 RANS 编码器（Range Asymmetric Numeral Systems）将这些量化值编码为一个字符串。
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z_hat.size()[-2:]}
    """
    encode: 编码函数
    为什么对于y并非直接进行量化？而先padding再量化？
    encode函数的目的是将输入图像编码为一组字符串，在这里我们就更加注重于他的压缩效率，我们追求的就是去最大程度去压缩图像
    encode函数借助compress_ar函数实现自回归压缩，从compressar的实现可以看到，他是通过对图像的每一个像素进行上下文的特征提取，
    对y进行padding，边缘像素也能获取完整邻域信息，这样在使用 MaskConv模块进行上下文预测时，能更准确估计每个像素的概率分布，为熵编码提供更精确参数，提高压缩效率。
    """
    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        """_summary_
        整体功能介绍
        _compress_ar 函数的作用是实现自回归（Auto-Regressive）压缩。它通过上下文预测模块和熵参数模块，对输入的特征图 y_hat 进行逐像素的压缩。
        该函数利用高斯条件模块计算每个像素的量化值，并通过 RANS 编码器（Range Asymmetric Numeral Systems）将这些量化值编码为一个字符串。
        
        参数分析：
        y_hat：填充后的特征图张量，维度为 (batch_size, M, height + 2*padding, width + 2*padding)。
        params：超先验模块生成的参数张量，维度为 (batch_size, M*2, height/32, width/32)。
        height：编码特征图的高度。
        width：编码特征图的宽度。
        kernel_size：上下文预测模块的卷积核大小。
        padding：填充大小。
        
        返回值：
        返回一个字符串，表示压缩后的特征图。
        
        整体执行逻辑:
        - 提取高斯条件模块的相关参数(cdf, cdf_length, offsets等)
        - 初始化 RANS 编码器。
        - 逐像素遍历特征图，对每个像素进行上下文预测和熵参数计算。 
        具体来说，我们遍历每一个像素位置，然后考虑该像素附近的一个小片(5x5的区域)，这个小片称为这个像素的“上下文”，对这个小片做一个二维卷积，得到一个1x1的“上下文预测值”，作为当前像素位置的上下文信息 ctx_p；
        另一方面，使用超先验解码器得到的处理后超先验参数p; 将ctx_p 和 p 拼接起来，得到拼接后特征 g= [ctx_p , p ]
        - 将拼接后特征g 输入给 self.entropy_parameters 熵参数预测网络， 得到预测的熵参数 gaussian_params，再将gaussian_params拆分为均值参数 means_hat 和 方差参数 scales_hat
        - 使用高斯条件模块对每个像素进行量化得到y_q
        - 将量化值编码为码流字符串。
        """
        
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        #* 提取高斯条件模块的累积分布函数（CDF）、CDF长度和偏移量。
        # 变量维度：
        # cdf 是一个列表，表示量化后变量的 CDF。
        # cdf_lengths 是一个列表，表示每个 CDF 的长度。
        # offsets 是一个列表，表示偏移量。


        encoder = BufferedRansEncoder() #* 初始化 RANS 编码器，
        symbols_list = []
        indexes_list = []
        # 并创建用于存储符号和索引的列表。
        # 变量维度：
        # encoder 是一个 BufferedRansEncoder 对象。
        # symbols_list 和 indexes_list 是空列表，用于存储编码过程中的符号和索引。

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask   #* 带有掩码的权重
        #! 注意： 尽管_compress_ar的参数y_hat在输入时不是量化版本(i.e. 非量化隐变量y)，但是后通过下面的循环将y_hat变为量化后隐变量hat{y}
        #* 并且保证，对于每一个空间坐标(h,w)，y_hat[:][:][h][w]在被用到之前，都已经被更新为量化版本
        #* 因此，本质上这里的y_hat就是"量化后隐变量hat{y}"
        for h in range(height):  
            for w in range(width):
                #* 遍历特征图y_hat的每个像素位置 (h, w), 下面我们要来预测 y_hat的(h+padding,w+padding)这个位置的像素
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]  #* 对每个像素位置，提取大小为 kernel_size x kernel_size 的局部特征图 y_crop
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )   #* 使用上下文预测模块的权重和偏置对 y_crop 进行卷积操作(F.conv2d)，得到上下文预测值 ctx_p
                #* ctx_p 表示 使用(h+padding,w+padding)的左上角那些已经解码了的像素来预测当前位置的像素的结果
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1] #* 提取超先验解码器处理后的参数 p，
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1)) #* 将参数 p和上下文预测值ctx_p拼接后，通过熵参数模块计算高斯分布的参数。
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                means_hat, scales_hat = gaussian_params.chunk(2, 1) #* 将高斯分布参数分离为均值 means_hat 和尺度 scales_hat
                #* 变量维度分析：
                # p 的维度为 (batch_size, M*2, 1, 1)。
                # gaussian_params 的维度为 (batch_size, M*2, 1, 1)。
                # means_hat 和 scales_hat 的维度均为 (batch_size, M)。
                indexes = self.gaussian_conditional.build_indexes(scales_hat)    #* 使用高斯条件模块构建索引 indexes。
                y_crop = y_crop[:, :, padding, padding] # 提取中心像素值 y_crop。
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)  #* 使用高斯条件模块对中心像素值进行量化，得到量化值 y_q。
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat #* 更新特征图 y_hat 的(h+padding, w+padding)位置，
                #! i.e. y_hat的(h+padding, w+padding)位置已经被更新量化后版本了，它还尚未被用到，在用到它的时候他已经是代表 量化后隐变量hat{y}

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )   #* 使用 RANS 编码器将符号和索引编码为字符串。
        string = encoder.flush()    #* string 是一个字节序列，表示压缩后的特征图。
        return string

    def decode(self, strings, shape):
        """_summary_
        1. 功能概述
        decode 方法是一个解码函数，用于从编码的字符串（通常是字节序列）中恢复原始数据。
        
        2. 参数分析
        strings：一个包含编码数据的列表，通常包含两个元素：
        - strings[0]：编码后的数据，用于解码。
        - strings[1]：辅助数据，例如超参数或上下文信息。
        shape：解码后数据的形状，通常是一个元组，表示解码后张量的维度。
        
        3. 返回值
        返回解码后的张量 y_hat(量化后隐特征)，y_hat 的最终形状为 (batch_size, channels, y_height, y_width)
        
        4. 整体执行逻辑
        decode 方法的主要逻辑包括：
        验证输入：确保 strings 是一个包含两个元素的列表。
        解压超参数：从 strings[1] 中解压超参数。
        初始化解码张量：根据 shape 初始化解码张量 y_hat，并进行适当的填充。
        逐元素解码：使用自回归解码器逐元素解码 strings[0], 将解码得到的内容填到y_hat(量化后隐特征)
        去除填充：去除解码张量的填充部分，恢复原始形状。
        """
        assert isinstance(strings, list) and len(strings) == 2  #* 确保 strings 是一个包含两个元素的列表。如果不是，抛出 AssertionError。

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder
        params, z_hat = self.hyperprior.decompress(strings[1], shape)   #* 调用 self.hyperprior.decompress 方法，从 strings[1] 中解压超参数 params 和 z_hat。
        # z_hat 是解压后的张量，形状由 shape 参数决定。

        
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2    #* 填充padding == 2

        #* 计算 y_hat 的高度和宽度，根据 z_hat 的形状和缩放因子 s。
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        #* 初始化 y_hat 为零张量，并添加适当的填充，以便可以直接处理子张量。
        #* y_hat 的形状为 (batch_size, channels, y_height + 2 * padding, y_width + 2 * padding)

        for i, y_string in enumerate(strings[0]):   #* 遍历 strings[0] 中的每个编码字符串 y_string。
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],   #* y_hat[i] 表示y_hat当前小批量中的第i个样本，y_hat形状为(C,H,W)
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )   #* 调用 _decompress_ar 方法，逐元素解码 y_string，并将结果存储在 y_hat 中。

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))  #* 使用 F.pad 方法去除 y_hat 的填充部分，恢复原始形状。
        #* y_hat 的最终形状为 (batch_size, channels, y_height, y_width)
        
        #x_hat = self.g_s(y_hat).clamp_(0, 1)
        return y_hat

    def _decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding):
        """_summary_
        1. 功能概述
        _decompress_ar 是一个自回归解码函数，用于逐像素解码输入的字节流 y_string，并恢复原始张量 y_hat。
        它利用上下文预测和高斯条件模型来解码每个像素值。这种方法通常用于图像压缩和视频编码中的熵解码。
        
        2. 参数分析
        y_string：编码后的字节流，用于解码。
        y_hat：初始化为零的张量，用于存储解码后的结果。
        params：超参数张量，包含解码所需的额外信息。
        height：解码后张量的高度。
        width：解码后张量的宽度。
        kernel_size：上下文预测卷积核的大小。
        padding：填充大小，用于处理边界情况。
        
        3. 返回值
        该方法没有返回值，但会直接修改输入张量 y_hat，将解码后的值填充到其中。
        
        4. 整体执行逻辑
        _decompress_ar 的主要逻辑包括：
        - 初始化解码器：创建 RansDecoder 实例并设置输入字节流。
        - 逐像素解码：通过上下文预测和高斯条件模型逐像素解码输入字节流。
        - 更新解码张量：将解码后的值填充到 y_hat 中。
        """
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder() #* 创建一个 RansDecoder 实例，用于解码输入字节流。
        decoder.set_stream(y_string)    #* 调用 set_stream 方法，将输入字节流 y_string 设置为解码器的输入。

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        #* 下面这个循环，用(h,w)位置一个5x5的小片作为“上下文”来预测 (h+2,w+2)位置的y_hat
        for h in range(height):
            for w in range(width):
                #* 遍历每个像素位置 (h, w)
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]  #* 从 y_hat 中裁剪出以 (h, w) 为中心的 kernel_size x kernel_size 的子张量 y_crop。
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )   #* 使用上下文预测卷积层 self.context_prediction 对 y_crop 进行卷积，得到上下文预测值 ctx_p。
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]  #* 从 params 中提取以 (h, w) 为中心的 1x1 子张量 p
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1)) #* 将 p 和 ctx_p 拼接起来，通过熵参数预测网络 self.entropy_parameters 得到高斯参数 means_hat 和 scales_hat。
                means_hat, scales_hat = gaussian_params.chunk(2, 1)
                #* means_hat 表示 高斯分布的均值， scales_hat 表示高斯分布的方差
                indexes = self.gaussian_conditional.build_indexes(scales_hat)   #* 使用 self.gaussian_conditional.build_indexes 构建索引 indexes
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )   #* 调用 decoder(RansDecoder).decode_stream 解码当前像素值 rv(i.e. 量化后隐特征y_hat)
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)    #* 对rv进行反量化操作，rv = rv + means_hat

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv  #* 将解码后的值 rv 填充到 y_hat 的相应位置, i.e. 空间位置为 (h+padding, w+padding)

class LDMIC_checkboard(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=JointContextTransfer, training=False):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )
        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)
        self.atten_4 = decode_atten(N)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )
        
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.context_prediction = CheckMaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.M = M
        self.N = N
        if training:
            self.training_ctx_params_anchor = torch.zeros([8, self.M * 2, 16, 16]).cuda()

    def forward(self, x):
        x_left, x_right = x[0], x[1] #x.chunk(2, 1)
        y_left, y_right = self.encoder(x_left), self.encoder(x_right)
        

        y_left_ste, y_left_likelihoods = self.forward_entropy(y_left)
        y_right_ste, y_right_likelihoods = self.forward_entropy(y_right) 

        y_left, y_right = self.atten_3(y_left_ste, y_right_ste)
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        x_left_hat, x_right_hat = self.decoder_2(y_left), self.decoder_2(y_right)

        return {
            "x_hat": [x_left_hat, x_right_hat],
            "likelihoods": [y_left_likelihoods, y_right_likelihoods],
        }

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def forward_entropy(self, y):
        params, z_likelihoods = self.hyperprior(y)

        batch_size, _, y_height, y_width = y.size()
        # compress anchor
        if self.training:
            ctx_params_anchor = self.training_ctx_params_anchor[:batch_size]
        else:
            ctx_params_anchor = torch.zeros([batch_size, self.M * 2, y_height, y_width]).to(y.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        anchor = self.get_anchor(y_hat)
        ctx_params_non_anchor = self.context_prediction(anchor)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        scales_hat, means_hat = self.merge(scales_anchor, means_anchor, 
            scales_non_anchor, means_non_anchor)
        _, y_likelihoods = self.gaussian_conditional(y, scales=scales_hat, means=means_hat)
        y_ste = quantize_ste(y-means_hat) + means_hat

        return y_ste, {"y": y_likelihoods, "z": z_likelihoods}

    def merge(self, scales_anchor, means_anchor, scales_non_anchor, means_non_anchor, mask_type="A"):
        scales_hat = scales_anchor.clone()
        means_hat = means_anchor.clone()
        if mask_type == "A":
            scales_hat[:, :, 0::2, 0::2] = scales_non_anchor[:, :, 0::2, 0::2]
            scales_hat[:, :, 1::2, 1::2] = scales_non_anchor[:, :, 1::2, 1::2]
            means_hat[:, :, 0::2, 0::2] = means_non_anchor[:, :, 0::2, 0::2]
            means_hat[:, :, 1::2, 1::2] = means_non_anchor[:, :, 1::2, 1::2]
        else:
            scales_hat[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
        return scales_hat, means_hat

    def get_anchor(self, y_hat, mask_type="A"):
        y_anchor = y_hat.clone()
        if mask_type == "A":
            y_anchor[:, :, 0::2, 0::2] = 0
            y_anchor[:, :, 1::2, 1::2] = 0
        else:
            y_anchor[:, :, 0::2, 1::2] = 0
            y_anchor[:, :, 1::2, 0::2] = 0
        return y_anchor

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated

    def compress(self, x):
        x_left, x_right = x[0], x[1]
        left_dict = self.encode(x_left)
        right_dict = self.encode(x_right)
        return left_dict, right_dict

    def decompress(self, left_dict, right_dict):
        y_left_hat = self.decode(left_dict["strings"], left_dict["shape"])
        y_right_hat = self.decode(right_dict["strings"], right_dict["shape"])
        #print(y_left_hat[0, 0, 0, 0:10], y_right_hat[0, 0, 0, 0:10])
        y_left, y_right = self.atten_3(y_left_hat, y_right_hat)
        #print(y_left[0, 0, 0, 0:10], y_right[0, 0, 0, 0:10])
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        #print(y_left[0, 0, 0, 0:10], y_right[0, 0, 0, 0:10])
        x_left_hat, x_right_hat = self.decoder_2(y_left), self.decoder_2(y_right) #.clamp_(0, 1), self.decoder_2(y_right).clamp_(0, 1)
        
        return {
            "x_hat": [x_left_hat, x_right_hat],
        }   

    def encode(self, x):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        batch_size, channel, x_height, x_width = x.shape

        y = self.encoder(x)

        y_a = y[:, :, 0::2, 0::2]
        y_d = y[:, :, 1::2, 1::2]
        y_b = y[:, :, 0::2, 1::2]
        y_c = y[:, :, 1::2, 0::2]

        params, z_hat, z_strings = self.hyperprior.compress(y)

        anchor = torch.zeros_like(y).to(x.device)
        anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, x_height // 16, x_width // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_strings = self.gaussian_conditional.compress(y_b, indexes_b, means_b)
        y_b_quantized = self.gaussian_conditional.decompress(y_b_strings, indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_strings = self.gaussian_conditional.compress(y_c, indexes_c, means_c)
        y_c_quantized = self.gaussian_conditional.decompress(y_c_strings, indexes_c, means=means_c)

        anchor_quantized = torch.zeros_like(y).to(x.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_strings = self.gaussian_conditional.compress(y_a, indexes_a, means=means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_strings = self.gaussian_conditional.compress(y_d, indexes_d, means=means_d)

        return {
            "strings": [y_a_strings, y_b_strings, y_c_strings, y_d_strings, z_strings],
            "shape": z_hat.size()[-2:]
        }

    def decode(self, strings, shape):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        params, z_hat = self.hyperprior.decompress(strings[4], shape)
        #z_hat = self.entropy_bottleneck.decompress(strings[4], shape)
        #params = self.h_s(z_hat)

        batch_size, channel, z_height, z_width = z_hat.shape
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, z_height * 4, z_width * 4]).to(z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_quantized = self.gaussian_conditional.decompress(strings[1], indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_quantized = self.gaussian_conditional.decompress(strings[2], indexes_c, means=means_c)

        anchor_quantized = torch.zeros([batch_size, self.M, z_height * 4, z_width * 4]).to(z_hat.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_quantized = self.gaussian_conditional.decompress(strings[0], indexes_a, means=means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_quantized = self.gaussian_conditional.decompress(strings[3], indexes_d, means=means_d)

        # Add non_anchor_quantized
        anchor_quantized[:, :, 0::2, 0::2] = y_a_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 1::2] = y_d_quantized[:, :, :, :]

        #print(anchor_quantized[0, 0, 0, :])
        return anchor_quantized 

class Multi_LDMIC(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=Multi_JointContextTransfer):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )

        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)
        self.atten_4 = decode_atten(N)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )
        
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.context_prediction = MaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        """_summary_

        输入: 
        x是一个长度为 self.num_camera 的 list; x[i] 是一个维度为(batch_size, C,H,W)的张量，表示第i个视点的这个小批量图像

        """
        num_camera = len(x) #* 获得 视点数

        x = torch.cat(x, dim=0) #* 按照第一维进行拼接
        #* x 的维度为: (batch_size*num_camera, C,H,W)
        y = self.encoder(x)
        params, z_likelihoods = self.hyperprior(y)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat([params, ctx_params], 1))
        means_hat, scales_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_ste = quantize_ste(y-means_hat)+means_hat

        y_ste = self.decoder_1(self.atten_3(y_ste, num_camera))
        x_hat = self.decoder_2(self.atten_4(y_ste, num_camera))
        
        x_hat_list = x_hat.chunk(num_camera, 0)
        z_likelihoods_list = z_likelihoods.chunk(num_camera, 0)
        y_likelihoods_list = y_likelihoods.chunk(num_camera, 0)
        likelihoods = [{"y": y_likelihood, "z": z_likelihood} for y_likelihood, z_likelihood in zip(y_likelihoods_list, z_likelihoods_list)]

        return {
            "x_hat": x_hat_list,
            "likelihoods": likelihoods,
        }

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def fix_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.hyperprior.parameters():
            p.requires_grad = False
        for p in self.context_prediction.parameters():
            p.requires_grad = False
        for p in self.entropy_parameters.parameters():
            p.requires_grad = False
        for p in self.gaussian_conditional.parameters():
            p.requires_grad = False

    def load_encoder(self, current_state_dict, checkpoint):
        encoder_dict = {k.replace("g_a", "encoder"): v for k, v in checkpoint.items() if "g_a" in k}
        context_prediction_dict = {k: v for k, v in checkpoint.items() if "context_prediction" in k}
        entropy_parameters_dict = {k: v for k, v in checkpoint.items() if "entropy_parameters" in k}
        gaussian_conditional_dict = {k: v for k, v in checkpoint.items() if "gaussian_conditional" in k}
        hyperprior_dict = {}
        for k, v in checkpoint.items():
            if "h_a" in k:
                hyperprior_dict[k.replace("h_a", "hyperprior.hyper_encoder")] = v
            elif "h_s" in k:
                hyperprior_dict[k.replace("h_s", "hyperprior.hyper_decoder")] = v
            elif "entropy_bottleneck" in k:
                hyperprior_dict[k.replace("entropy_bottleneck", "hyperprior.entropy_bottleneck")] = v

        current_state_dict.update(encoder_dict)
        current_state_dict.update(hyperprior_dict)
        current_state_dict.update(context_prediction_dict)
        current_state_dict.update(entropy_parameters_dict)
        current_state_dict.update(gaussian_conditional_dict)
        return current_state_dict

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated

class Multi_LDMIC_checkboard(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=Multi_JointContextTransfer, training=False):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )
    
        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)
        self.atten_4 = decode_atten(N)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )
        
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.context_prediction = CheckMaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.M = M

        if training:
            self.training_ctx_params_anchor = torch.zeros([8*7, M * 2, 16, 16]).cuda()

    def forward(self, x):
        num_camera = len(x)
        x = torch.cat(x, dim=0)
        y = self.encoder(x)

        y_ste, y_likelihoods, z_likelihoods = self.forward_entropy(y)
        y_ste = self.decoder_1(self.atten_3(y_ste, num_camera))
        x_hat = self.decoder_2(self.atten_4(y_ste, num_camera))
        
        x_hat_list = x_hat.chunk(num_camera, 0)
        z_likelihoods_list = z_likelihoods.chunk(num_camera, 0)
        y_likelihoods_list = y_likelihoods.chunk(num_camera, 0)
        likelihoods = [{"y": y_likelihood, "z": z_likelihood} for y_likelihood, z_likelihood in zip(y_likelihoods_list, z_likelihoods_list)]

        return {
            "x_hat": x_hat_list,
            "likelihoods": likelihoods,
        }

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list


    def forward_entropy(self, y):
        params, z_likelihoods = self.hyperprior(y)

        batch_size, _, y_height, y_width = y.size()
        # compress anchor
        if self.training:
            ctx_params_anchor = self.training_ctx_params_anchor[:batch_size]
        else:
            ctx_params_anchor = torch.zeros([batch_size, self.M * 2, y_height, y_width]).to(y.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        anchor = self.get_anchor(y_hat)
        ctx_params_non_anchor = self.context_prediction(anchor)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        scales_hat, means_hat = self.merge(scales_anchor, means_anchor, 
            scales_non_anchor, means_non_anchor)
        _, y_likelihoods = self.gaussian_conditional(y, scales=scales_hat, means=means_hat)
        y_ste = quantize_ste(y-means_hat)+means_hat

        return y_ste, y_likelihoods, z_likelihoods

    def merge(self, scales_anchor, means_anchor, scales_non_anchor, means_non_anchor, mask_type="A"):
        scales_hat = scales_anchor.clone()
        means_hat = means_anchor.clone()
        if mask_type == "A":
            scales_hat[:, :, 0::2, 0::2] = scales_non_anchor[:, :, 0::2, 0::2]
            scales_hat[:, :, 1::2, 1::2] = scales_non_anchor[:, :, 1::2, 1::2]
            means_hat[:, :, 0::2, 0::2] = means_non_anchor[:, :, 0::2, 0::2]
            means_hat[:, :, 1::2, 1::2] = means_non_anchor[:, :, 1::2, 1::2]
        else:
            scales_hat[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
        return scales_hat, means_hat

    def get_anchor(self, y_hat, mask_type="A"):
        y_anchor = y_hat.clone()
        if mask_type == "A":
            y_anchor[:, :, 0::2, 0::2] = 0
            y_anchor[:, :, 1::2, 1::2] = 0
        else:
            y_anchor[:, :, 0::2, 1::2] = 0
            y_anchor[:, :, 1::2, 0::2] = 0
        return y_anchor

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated
 

class Multi_MSE_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, device):
        super().__init__()
        self.mse = nn.MSELoss().to(device)
        #self.num_camera = num_camera
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        #target1, target2 = target[0], target[1]
        num_camera = len(target)
        N, _, H, W = target[0].size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = 0
        out["mse_loss"] = 0
        out["psnr"] = 0

        # 计算误差
        for i in range(num_camera):
            out['bpp'+str(i)] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output['likelihoods'][i].values())
            out["mse"+str(i)] = self.mse(output['x_hat'][i], target[i])
            out["bpp_loss"] += out['bpp'+str(i)]/num_camera
            out['mse_loss'] += lmbda * out["mse"+str(i)] /num_camera
            if out["mse"+str(i)] > 0:
                out["psnr"+str(i)] = 10 * (torch.log10(1 / out["mse"+str(i)])).mean()
            else:
                out["psnr"+str(i)] = 0
            out["psnr"] += out["psnr"+str(i)]/num_camera
        
        out['loss'] = out['mse_loss'] + out['bpp_loss']
        return out

class Multi_MS_SSIM_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, device, size_average=True, max_val=1):
        super().__init__()
        self.ms_ssim = MS_SSIM(size_average, max_val).to(device)
        #self.num_camera = num_camera
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        #target1, target2 = target[0], target[1]
        num_camera = len(target)
        N, _, H, W = target[0].size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = 0
        out["ms_ssim_loss"] = 0
        out["ms_db"] = 0

        # 计算误差
        for i in range(num_camera):
            out['bpp'+str(i)] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output['likelihoods'][i].values())
            out["ms_ssim"+str(i)] = 1 - self.ms_ssim(output['x_hat'][i], target[i])
            out["bpp_loss"] += out['bpp'+str(i)]/num_camera
            out['ms_ssim_loss'] += lmbda * out["ms_ssim"+str(i)] /num_camera
            if out["ms_ssim"+str(i)] > 0:
                out["ms_db"+str(i)] = 10 * (torch.log10(1 / out["ms_ssim"+str(i)])).mean()
            else:
                out["ms_db"+str(i)] = 0
            out["ms_db"] += out["ms_db"+str(i)]/num_camera
        
        out['loss'] = out['ms_ssim_loss'] + out['bpp_loss']
        return out

class MSE_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        target1, target2 = target[0], target[1]
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())        
        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2
        out["mse0"] = self.mse(output['x_hat'][0], target1)
        out["mse1"] = self.mse(output['x_hat'][1], target2)
        
        if isinstance(lmbda, list):
            out['mse_loss'] = (lmbda[0] * out["mse0"] + lmbda[1] * out["mse1"])/2 
        else:
            out['mse_loss'] = lmbda * (out["mse0"] + out["mse1"])/2        #end to end
        out['loss'] = out['mse_loss'] + out['bpp_loss']

        return out

class MS_SSIM_Loss(nn.Module):
    def __init__(self, device, size_average=True, max_val=1):
        super().__init__()
        self.ms_ssim = MS_SSIM(size_average, max_val).to(device)
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        target1, target2 = target[0], target[1]
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())        
        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2

        out["ms_ssim0"] = 1 - self.ms_ssim(output['x_hat'][0], target1)
        out["ms_ssim1"] = 1- self.ms_ssim(output['x_hat'][1], target2)
 
        out['ms_ssim_loss'] = (out["ms_ssim0"] + out["ms_ssim1"])/2        #end to end
        out['loss'] = lmbda * out['ms_ssim_loss'] + out['bpp_loss']
        return out


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

class MS_SSIM(nn.Module):
    def __init__(self, size_average=True, max_val=255, device_id=0):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val
        self.device_id = device_id

    def _ssim(self, img1, img2):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11

        window = create_window(window_size, sigma, self.channel)
        if self.device_id != None:
            window = window.cuda(self.device_id)

        mu1 = F.conv2d(img1, window, padding=window_size //
                       2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size //
                       2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(
            img1*img1, window, padding=window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2*img2, window, padding=window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size //
                           2, groups=self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if self.size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))
        msssim=Variable(torch.Tensor(levels,))
        mcs=Variable(torch.Tensor(levels,))
        # if self.device_id != None:
        #     weight = weight.cuda(self.device_id)
        #     weight = msssim.cuda(self.device_id)
        #     weight = mcs.cuda(self.device_id)
        #     print(weight.device)

        for i in range(levels):
            ssim_map, mcs_map=self._ssim(img1, img2)
            msssim[i]=ssim_map
            mcs[i]=mcs_map
            filtered_im1=F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2=F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1=filtered_im1
            img2=filtered_im2

        value=(torch.prod(mcs[0:levels-1]**weight[0:levels-1]) *
                                    (msssim[levels-1]**weight[levels-1]))
        return value


    def forward(self, img1, img2, levels=5):
        return self.ms_ssim(img1, img2, levels)



