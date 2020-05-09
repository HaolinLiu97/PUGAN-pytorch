import argparse
import os, sys
sys.path.append("../")

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument("--model", type=str, default='punet')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument('--up_ratio', type=int, default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument("--use_bn", action='store_true', default=False)
parser.add_argument("--use_res", action='store_true', default=False)
parser.add_argument('--resume', type=str, required=True)

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from data.data_loader import PUNET_Dataset
from chamfer_distance import chamfer_distance
from auction_match import auction_match
import pointnet2.utils.pointnet2_utils as pn2_utils
import importlib
from network.networks import Generator
from option.train_option import get_train_options


def get_emd_loss(pred, gt, pcd_radius):
    idx, _ = auction_match(pred, gt)
    matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
    matched_out = matched_out.transpose(1, 2).contiguous()
    dist2 = (pred - matched_out) ** 2
    dist2 = dist2.view(dist2.shape[0], -1)  # <-- ???
    dist2 = torch.mean(dist2, dim=1, keepdims=True)  # B,
    dist2 /= pcd_radius
    return torch.mean(dist2)


def get_cd_loss(pred, gt, pcd_radius):
    cost_for, cost_bac = chamfer_distance(gt, pred)
    cost = 0.5 * cost_for + 0.5 * cost_bac
    cost /= pcd_radius
    cost = torch.mean(cost)
    return cost


if __name__ == '__main__':
    param=get_train_options()
    model = Generator()

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
    model.eval().cuda()

    eval_dst = PUNET_Dataset(h5_file_path='../Patches_noHole_and_collected.h5', split_dir=param['test_split'], isTrain=False)
    eval_loader = DataLoader(eval_dst, batch_size=args.batch_size,
                             shuffle=False, pin_memory=True, num_workers=args.workers)

    emd_list = []
    cd_list = []
    with torch.no_grad():
        for itr, batch in enumerate(eval_loader):
            points, gt, radius = batch
            points = points[..., :3].permute(0,2,1).float().cuda().contiguous()
            gt = gt[..., :3].float().cuda().contiguous()
            radius = radius.float().cuda()
            preds = model(points)  # points.shape[1])
            preds=preds.permute(0,2,1).contiguous()

            emd = get_emd_loss(preds, gt, radius)
            cd = get_cd_loss(preds, gt, radius)
            print(' -- iter {}, emd {}, cd {}.'.format(itr, emd, cd))
            emd_list.append(emd.item())
            cd_list.append(cd.item())

    print('mean emd: {}'.format(np.mean(emd_list)))
    print('mean cd: {}'.format(np.mean(cd_list)))