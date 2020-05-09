import argparse
import os, sys
sys.path.append("../")

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--resume', type=str, required=True)
parser.add_argument('--exp_name',type=str,required=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.xyz_util import save_xyz_file

from network.networks import Generator
from data.data_loader import PUNET_Dataset_Whole

if __name__ == '__main__':
    model = Generator()

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
    model.eval().cuda()

    eval_dst = PUNET_Dataset_Whole(data_dir='../MC_5k')
    eval_loader = DataLoader(eval_dst, batch_size=1,
                             shuffle=False, pin_memory=True, num_workers=0)

    names = eval_dst.names
    exp_name=args.exp_name
    save_dir=os.path.join('../outputs',exp_name)
    if os.path.exists(save_dir)==False:
        os.makedirs(save_dir)
    for itr, batch in enumerate(eval_loader):
        name = names[itr]
        points = batch[:,:,0:3].permute(0,2,1).float().cuda()
        preds = model(points)
        #radius=radius.float().cuda()
        #centroid=centroid.float().cuda()
        #print(preds.shape,radius.shape,centroid.shape)
        #preds=preds*radius+centroid.unsqueeze(2).repeat(1,1,4096)

        preds = preds.permute(0,2,1).data.cpu().numpy()[0]
        points = points.permute(0,2,1).data.cpu().numpy()
        save_file='../outputs/{}/{}.xyz'.format(exp_name,name)
        #print(preds.shape)
        save_xyz_file(preds,save_file)

