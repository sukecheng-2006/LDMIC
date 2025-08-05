import argparse
import math
import random
import sys
import time

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from models.ldmic import *
from lib.utils import get_output_folder, AverageMeter, save_checkpoint, MultiCameraImageDataset, StereoImageDataset
import numpy as np

import yaml
import wandb
import os
from tqdm import tqdm
from pytorch_msssim import ms_ssim
from datetime import datetime
os.environ["WANDB_API_KEY"] = "8c54ec5e7bd79af9529a1ad26f54f7505fbe9470" # long feiyue API 
#os.environ["WANDB_API_KEY"] = "3ee0a4687850e1410055cec0cc8d52f376ad146b" # zht API

def compute_aux_loss(aux_list: List, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss
        if backward is True:
            aux_loss.backward()

    return aux_loss_sum

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(p for p in net.named_parameters() if p[1].requires_grad)
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, args):
    model.train()
    device = next(model.parameters()).device
    if args.metric == "mse":
        metric_dB_name = 'psnr'
        metric_name = "mse_loss" 
    else:
        metric_dB_name = "ms_db"
        metric_name = "ms_ssim_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')  

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')

    print('Train epoch:', epoch)
    with tqdm(total=len(train_dataloader)) as loop:
        for i, batch in enumerate(train_dataloader):
            
            #* 对于一个小批量 batch, 他是一个长度为 self.num_camera 的 list; batch[i] 是一个维度为(batch_size, C,H,W)的张量，表示第i个视点的这个小批量图像
            d = [frame.to(device) for frame in batch]   #* 将小批量加载到制定设备商
            #* d是一个长度为 self.num_camera 的 list; d[i] 是一个维度为(batch_size, C,H,W)的张量，表示第i个视点的这个小批量图像
            optimizer.zero_grad()
            if aux_optimizer is not None:
                aux_optimizer.zero_grad()
        
            out_net = model(d)
            out_criterion = criterion(out_net, d, args.lmbda)

            out_criterion["loss"].backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

            if aux_optimizer is not None:
                out_aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
                aux_optimizer.step()
            else:
                out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)

            loss.update(out_criterion["loss"].item())
            bpp_loss.update((out_criterion["bpp_loss"]).item())
            aux_loss.update(out_aux_loss.item())
            metric_loss.update(out_criterion[metric_name].item())
            metric_dB.update(out_criterion[metric_dB_name].item())


            loop.set_description('[{}/{}]'.format(i, len(train_dataloader)))
            loop.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
                metric_dB_name:metric_dB.avg})
            loop.update(1)
    torch.cuda.empty_cache()  
    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg, }

    return out

def test_epoch(epoch, val_dataloader, model, criterion, args):
    model.eval()
    device = next(model.parameters()).device

    if args.metric == "mse":
        metric_dB_name = 'psnr'
        metric_name = "mse_loss" 
    else:
        metric_dB_name = "ms_db"
        metric_name = "ms_ssim_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')  

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    loop = tqdm(val_dataloader)

    with torch.no_grad():
        for i, batch in enumerate(loop):
            d = [frame.to(device) for frame in batch]
              
            out_net = model(d)
            out_criterion = criterion(out_net, d, args.lmbda)
            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)

            loss.update(out_criterion["loss"].item())
            bpp_loss.update((out_criterion["bpp_loss"]).item())
            aux_loss.update(out_aux_loss.item())
            metric_loss.update(out_criterion[metric_name].item())
            metric_dB.update(out_criterion[metric_dB_name].item())

            loop.set_description('[{}/{}]'.format(i, len(val_dataloader)))
            loop.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
                metric_dB_name:metric_dB.avg})
    torch.cuda.empty_cache()
    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg, }

    return out


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--data_path", type=str,
        default='D:/researchdata',
        #default='../researchdata',
        help="The root path of dataset"
    )
    parser.add_argument(
        "--data_name", type=str, 
        #default='Mip-NeRF360', 
        default='wildtrack', 
        # default='Instereo2K', 
        #default='cityscape',
        help="The name of dataset"
    )
    parser.add_argument(
        "--model-name", type=str, default='Multi_LDMIC', help="Training dataset"
    )
    parser.add_argument(
        "--num_camera", type=int, default=2, help="The number of cameras"
    )
    parser.add_argument(
        "--dir_num", type=int, default=2, help="The number of dir in the data_path"
    )
    parser.add_argument(
        "--train_test_ratio", type=int, default=20, help="The ratio of the number of traindata to testdata"
    )   
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=2,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=2048,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--GPU_ID", type=int, default=0, help="GPU id")
    parser.add_argument(
        "--save", action="store_true", help="Save model to disk"
    )
    parser.add_argument(
        "--resize", action="store_true", help="training use resize or randomcrop"
    )
    parser.add_argument(
        "--seed", type=float, default=1, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--i_model_path", type=str, help="Path to a checkpoint")
    parser.add_argument("--metric", type=str, default="mse", help="metric: mse, ms-ssim")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
 
    if(args.data_name == 'Instereo2K' or args.data_name == 'cityscape'):
        train_dataset = StereoImageDataset(ds_type='train', root=args.data_path, ds_name =args.data_name, crop_size=args.patch_size, train_test_ratio = args.train_test_ratio)  
        test_dataset = StereoImageDataset(ds_type='test', root=args.data_path, ds_name =args.data_name, crop_size=args.patch_size, train_test_ratio = args.train_test_ratio) 
    else:
        train_dataset = MultiCameraImageDataset(ds_type='train', root=args.data_path, ds_name =args.data_name, crop_size=args.patch_size, num_camera=args.num_camera,  
                                                dir_num= args.dir_num, train_test_ratio = args.train_test_ratio)
        
        test_dataset = MultiCameraImageDataset(ds_type='test', root=args.data_path, ds_name =args.data_name, crop_size=args.patch_size, num_camera=args.num_camera, 
                                            dir_num= args.dir_num, train_test_ratio = args.train_test_ratio) 
    #* train_dataset是多视点数据集的训练数据集; 
    #* AdaptiveMultiCameraImageDataset 的__getitem__ 会返回一个长度为 self.num_camera 的 list: my_list; my_list[i] 是一个维度为(C,H,W)的张量，表示第i个视点的图像
    #! 由于小批量处理，一个批量的东西会在维度C之前添加一个batch_size 维度; 也就是说，对于一个小批量 batch, 他是一个长度为 self.num_camera 的 list; batch[i] 是一个维度为(batch_size, C,H,W)的张量，表示第i个视点的这个小批量图像

    device = f"cuda:{args.GPU_ID}" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
        shuffle=True, pin_memory=(device == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, 
        shuffle=False, pin_memory=(device == "cuda"))

    if args.model_name == "Multi_LDMIC":
        net = Multi_LDMIC(N=192, M=192, decode_atten=Multi_JointContextTransfer,)
    elif args.model_name == "Multi_LDMIC_checkboard":
        net = Multi_LDMIC_checkboard(N=192, M=192, decode_atten=Multi_JointContextTransfer,)
        
    net = net.to(device) 

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300, 400], 0.5)
    if args.metric == "mse":
        criterion = Multi_MSE_Loss(device) #MSE_Loss(lmbda=args.lmbda)
    else:
        criterion = Multi_MS_SSIM_Loss(device) #(device, lmbda=args.lmbda)
    last_epoch = 0
    best_loss = float("inf")

    if args.i_model_path:  # load from previous checkpoint
        print("Loading model: ", args.i_model_path)
        checkpoint = torch.load(args.i_model_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        last_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        best_b_model_path = os.path.join(os.path.split(args.i_model_path)[0], 'ckpt.best.pth.tar')
        best_loss = torch.load(best_b_model_path)["loss"]



    now = datetime.now()
    # 将日期和时间转换为字符串
    # 格式可以根据需要进行调整
    date_time_str = now.strftime("%Y-%m-%d-%H:%M:%S")
    log_dir, experiment_id = get_output_folder('./checkpoints/{}/{}/{}/lamda{}/'.format(args.data_name, args.metric, args.model_name, int(args.lmbda)), 'train')
    display_name = "{}_num_camera: {}_dir_num :{}_{}_lmbda{}".format(args.model_name,args.num_camera,args.dir_num, args.metric, int(args.lmbda))+date_time_str
    tags = "lmbda{}".format(args.lmbda)

    with open(os.path.join(log_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    project_name = "zht_imagecompression_" + args.data_name
    wandb.init(project=project_name, name=display_name, tags=[tags],) #notes="lmbda{}".format(args.lmbda))
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
    wandb.config.update(args) # config is a variable that holds and saves hyper parameters and inputs

    if args.metric == "mse":
        metric_dB_name = 'psnr'
        metric_name = "mse_loss" 
    else:
        metric_dB_name = "ms_db"
        metric_name = "ms_ssim_loss"

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss = train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, args)
        lr_scheduler.step()

        wandb.log({"train": {"loss": train_loss["loss"], metric_name: train_loss[metric_name], "bpp_loss": train_loss["bpp_loss"],
            "aux_loss": train_loss["aux_loss"], metric_dB_name: train_loss[metric_dB_name], }, }
        )
        loss = train_loss["loss"]
        if epoch%10==0:
            val_loss = test_epoch(epoch, test_dataloader, net, criterion, args)
            wandb.log({ 
                "test": {"loss": val_loss["loss"], metric_name: val_loss[metric_name], "bpp_loss": val_loss["bpp_loss"],
                "aux_loss": val_loss["aux_loss"], metric_dB_name: val_loss[metric_dB_name], }
                })
        
            loss = val_loss["loss"]
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
        else:
            loss = best_loss
            is_best = False
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    'lr_scheduler': lr_scheduler.state_dict(),
                },
                is_best, log_dir
            )

if __name__ == "__main__":
    main(sys.argv[1:])