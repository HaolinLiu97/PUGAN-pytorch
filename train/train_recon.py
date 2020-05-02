import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('../')
import torch
from network.networks import Generator, Discriminator,Generator_recon
from data.data_loader import PUNET_Dataset
import argparse
import time
from option.train_option import get_train_options
from utils.Logger import Logger
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from loss.loss import Loss
import datetime
import torch.nn as nn
from utils.visualize_utils import visualize_point_cloud
import numpy as np


def xavier_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def train(args):
    start_t = time.time()
    params = get_train_options()
    params["exp_name"] = args.exp_name
    params["patch_num_point"] = 1024
    params["batch_size"] = args.batch_size
    params['use_gan'] = args.use_gan

    if args.debug:
        params["nepoch"] = 2
        params["model_save_interval"] = 3
        params['model_vis_interval'] = 3

    log_dir = os.path.join(params["model_save_dir"], args.exp_name)
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    tb_logger = Logger(log_dir)

    trainloader = PUNET_Dataset(h5_file_path=params["dataset_dir"])
    # print(params["dataset_dir"])
    num_workers = 4
    train_data_loader = data.DataLoader(dataset=trainloader, batch_size=params["batch_size"], shuffle=True,
                                        num_workers=num_workers, pin_memory=True, drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G_model = Generator_recon(params)
    G_model.apply(xavier_init)
    G_model = torch.nn.DataParallel(G_model).to(device)
    D_model = torch.nn.DataParallel(Discriminator(params, in_channels=3)).to(device)

    G_model.train()
    D_model.train()

    optimizer_D = Adam(D_model.parameters(), lr=params["lr_D"], betas=(0.9, 0.999))
    optimizer_G = Adam(G_model.parameters(), lr=params["lr_G"], betas=(0.9, 0.999))

    D_scheduler = MultiStepLR(optimizer_D, [50, 80], gamma=0.2)
    G_scheduler = MultiStepLR(optimizer_G, [50, 80], gamma=0.2)

    Loss_fn = Loss()

    print("preparation time is %fs" % (time.time() - start_t))
    iter = 0
    for e in range(params["nepoch"]):
        D_scheduler.step()
        G_scheduler.step()
        for batch_id, (input_data, gt_data, radius_data) in enumerate(train_data_loader):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            input_data = input_data[:, :, 0:3].permute(0, 2, 1).float().cuda()
            gt_data = gt_data[:, :, 0:3].permute(0, 2, 1).float().cuda()

            start_t_batch = time.time()
            output_point_cloud = G_model(input_data)

            emd_loss = Loss_fn.get_emd_loss(output_point_cloud.permute(0, 2, 1), input_data.permute(0, 2, 1))

            total_G_loss=emd_loss
            total_G_loss.backward()
            optimizer_G.step()

            current_lr_D = optimizer_D.state_dict()['param_groups'][0]['lr']
            current_lr_G = optimizer_G.state_dict()['param_groups'][0]['lr']

            tb_logger.scalar_summary('emd_loss', emd_loss.item(), iter)
            tb_logger.scalar_summary('lr_D', current_lr_D, iter)
            tb_logger.scalar_summary('lr_G', current_lr_G, iter)

            msg = "{:0>8},{}:{}, [{}/{}], {}: {},{}:{}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(train_data_loader),
                "total_G_loss",
                total_G_loss.item(),
                "iter time",
                (time.time() - start_t_batch)
            )
            print(msg)

            if iter % params['model_save_interval'] == 0 and iter > 0:
                model_save_dir = os.path.join(params['model_save_dir'], params['exp_name'])
                if os.path.exists(model_save_dir) == False:
                    os.makedirs(model_save_dir)
                D_ckpt_model_filename = "D_iter_%d.pth" % (iter)
                G_ckpt_model_filename = "G_iter_%d.pth" % (iter)
                D_model_save_path = os.path.join(model_save_dir, D_ckpt_model_filename)
                G_model_save_path = os.path.join(model_save_dir, G_ckpt_model_filename)
                torch.save(D_model.module.state_dict(), D_model_save_path)
                torch.save(G_model.module.state_dict(), G_model_save_path)

            if iter % params['model_vis_interval'] == 0 and iter > 0:
                np_pcd = output_point_cloud.permute(0, 2, 1)[0].detach().cpu().numpy()
                # print(np_pcd.shape)
                img = (np.array(visualize_point_cloud(np_pcd)) * 255).astype(np.uint8)
                tb_logger.image_summary("images", img[np.newaxis, :], iter)

                gt_pcd = gt_data.permute(0, 2, 1)[0].detach().cpu().numpy()
                # print(gt_pcd.shape)
                gt_img = (np.array(visualize_point_cloud(gt_pcd)) * 255).astype(np.uint8)
                tb_logger.image_summary("gt", gt_img[np.newaxis, :], iter)

                input_pcd = input_data.permute(0, 2, 1)[0].detach().cpu().numpy()
                input_img = (np.array(visualize_point_cloud(input_pcd)) * 255).astype(np.uint8)
                tb_logger.image_summary("input", input_img[np.newaxis, :], iter)
            iter += 1


if __name__ == "__main__":
    import colored_traceback

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-e', type=str, required=True, help='experiment name')
    parser.add_argument('--debug', action='store_true', help='specify debug mode')
    parser.add_argument('--use_gan', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    train(args)