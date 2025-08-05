import argparse
import json
import math
import sys
import os
import time
import struct

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm
import compressai

from compressai.zoo.pretrained import load_pretrained
from compressai.zoo.image import model_urls, cfgs
from models.parahydra import *
from models.entropy_model import *
#from model_zoo import models_arch
from torch.hub import load_state_dict_from_url
from lib.util_bitstream import *
from lib.util_bitstream import pad as pad_util
from lib.set_logic import *
from lib.main_logic import *




def test_model(test_dataloader, model, save_dir, args):
    model.eval()
    device = next(model.parameters()).device

    psnr_log = AverageMeter('psnr', ':.4e')
    bpp_log = AverageMeter('bpp', ':.4e')
    msssim_log = AverageMeter('msssim', ':.4e')
    compress_time_log = AverageMeter('compress_cost_time', ':.4e')
    decompress_time_log = AverageMeter('decompress_cost_time', ':.4e')
    
    with torch.no_grad():
        with tqdm(total=len(test_dataloader)) as pbar:
            for i, img in enumerate(test_dataloader):
                x_list = [x.to(device) for x in img]
                B, C, H, W = x_list[0].shape
                                
                x_list = [
                pad_util(x, p = 64)
                for x in x_list
                ]
                if(i==0):   #* 对于第一张图片，预热模型，让模型充分进入GPU状态，然后再正式计时
                    bpp, compress_cost_time = compress_one_image(model=model, x=x_list, stream_path=save_dir, H=H, W=W, img_name=str(i),args = args)

                bpp, compress_cost_time = compress_one_image(model=model, x=x_list, stream_path=save_dir, H=H, W=W, img_name=str(i),args = args)
                #* bpp 是 num_camera 个图像一共所使用的码率, compress_cost_time 是 num_camera个图像一共所使用的编码时间
                print('end of compress_one_image')
                x_hat,  decompress_cost_time= decompress_one_image(model=model, stream_path=save_dir, img_name=str(i),args = args)
                #* decompress_cost_time 是 num_camera个图像一共所使用的解码时间
                print('end of decompress_one_image')
                
                
                
                metrics = {
                    "psnr-float": 0.0,
                    "ms-ssim-float": 0.0,
                    "bpp": 0.0,
                    "compress_cost_time":0.0,
                    "decompress_cost_time":0.0,
                }
                for idx in range(args.num_camera):
                    assert x_list[idx].dim() == 4 and x_hat[idx].dim() == 4, "Input must be (B, C, H, W)"
                    x_ori = x_list[idx][0]
                    x_rec = x_hat[idx][0]
                    rec = torch2img(x_rec)
                    ori = torch2img(x_ori)
                    ori.save(os.path.join(save_dir, f'C{idx}_{i}_gt.png' ))
                    rec.save(os.path.join(save_dir, f'C{idx}_{i}_rec.png' ))
                    metrics[f"index{idx}-psnr-float"], metrics[f"index{idx}-ms-ssim-float"] = compute_metrics(a=ori, b=rec)
                    # metrics[f"index{idx}-bpp"] = bpp
                    
                    metrics["psnr-float"] += metrics[f"index{idx}-psnr-float"]/args.num_camera
                    metrics["ms-ssim-float"] += metrics[f"index{idx}-ms-ssim-float"]/args.num_camera
                metrics["bpp"] += bpp / args.num_camera
                metrics["compress_cost_time"] += compress_cost_time / args.num_camera
                metrics["decompress_cost_time"] += decompress_cost_time / args.num_camera
                print(f'i is {i}, bpp is {metrics["bpp"]}, psnr is {metrics["psnr-float"]}, ms-ssim is {metrics["ms-ssim-float"]},\
                      compress time {metrics["compress_cost_time"]}s, decompress time {metrics["decompress_cost_time"]}s',flush = True)
                
                psnr_log.update(metrics["psnr-float"])
                bpp_log.update(metrics["bpp"])
                msssim_log.update(metrics["ms-ssim-float"])
                compress_time_log.update(metrics['compress_cost_time'])
                decompress_time_log.update(metrics['decompress_cost_time'])
                
                pbar.update(1)

    results = {
        "bpp" : bpp_log.avg, 
        "psnr" : psnr_log.avg, 
        "ms-ssim": msssim_log.avg,
        "compress_time": compress_time_log.avg,
        "decompress_time": decompress_time_log.avg,
    }

    return results





def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    if(args.use_reorder):
        reorder_file = set_reorder(args)
        sys.stdout = reorder_file    
    if args.seed is not None:
        set_seed(args.seed)    
    # 检查 test_batch_size 是否为 1
    if args.test_batch_size != 1:
        print(f"Error: test_batch_size must be 1, but got {args.test_batch_size}", file=sys.stderr)
        sys.exit(1)  # 非零退出码表示错误
    
    train_dataset, test_dataset, train_dataloader, test_dataloader = set_datasets(args)

    device = my_set_device(args)
    model = set_model(args)
    model = model.to(device) 
    optimizer, aux_optimizer = configure_optimizers(model, args)

    lr_scheduler, lr_scheduler_aux = LrScheduler(optimizer= optimizer, aux_optimizer = aux_optimizer, strategy = args.lr_strategy ,scale_factor=args.lr_scale_factor,
                                                 warmup_steps= args.warmup_steps, decay_factor = args.lr_decay_factor, interval_size = args.lr_interval_size, restart = args.restart,last_epoch=-1)
    
    mygradient_scheduler = gradient_scheduler(scale_facotr= args.gradient_scale_facotr, clip_max_norm = args.clip_max_norm if(args.clip_max_norm >0) else 1 ,
                                              scheduler_type=args.gradient_type, phase_length= args.phase_length)
    
    criterion = set_loss(device, args)


    if args.i_model_path:  # load from previous checkpoint

        checkpoint, model, last_epoch, optimizer, aux_optimizer,best_loss, lr_scheduler, lr_scheduler_aux  = load_my_model(args, device, model, optimizer, aux_optimizer,lr_scheduler, lr_scheduler_aux)

        model.update(force=True)  #! Attention: 这条语句一定要加，不要漏了


    # create output directory
    outputdir = args.output_eval
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    # results = defaultdict(list)
    args_dict = vars(args)
    print(args_dict)
    trained_net = f"{args.model_name}-{args.metric}"
    # metrics = run_inference(test_dataloader, model, **args_dict)
    results = test_model(test_dataloader= test_dataloader, model = model ,save_dir = args.output_eval, args = args)
    print(f'end of test_model')
    # for k, v in metrics.items():
    #     results[k].append(v)

    output = {
        "name": f"{args.model_name}-{args.metric}",
        "lambda" : f'{args.lmbda}',
        "data_name" : f'{args.data_name}',
        "num_camera":f'{args.num_camera}',
        "dir":f'{args.dir_num}',
        "results": results,
    }

    with (Path(f"{outputdir}/{args.model_name}_{args.data_name}_test_result.json")).open("wb") as f:
        f.write(json.dumps(output, indent=2).encode())
    print(json.dumps(output, indent=2))

    if args.use_reorder:
        # 恢复 sys.stdout
        sys.stdout = sys.__stdout__
        reorder_file.close()  

if __name__ == "__main__":
    main(sys.argv[1:])
