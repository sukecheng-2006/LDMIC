#
import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
# 获取父级文件夹路径
parent_directory = os.path.dirname(current_file_path)
# 将父级文件夹路径添加到系统路径中
sys.path.append(parent_directory)
from lib.utils import save_tensors_to_file


