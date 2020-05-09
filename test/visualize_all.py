import cv2
import argparse
import glob
import numpy as np
import os, sys
sys.path.append("../")
from utils.pc_util import draw_point_cloud

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--exp_name',type=str,required=True)

args = parser.parse_args()

if __name__=="__main__":
    file_dir=glob.glob(os.path.join(args.data_dir,"*.xyz"))##visualize all xyz file
    pcd_list=[]
    for file in file_dir:
        if file.split('/')[-1].find("_")<0:
            pcd_list.append(file)
    image_save_dir=os.path.join("../vis_result",args.exp_name)
    if os.path.exists(image_save_dir)==False:
        os.makedirs(image_save_dir)

    for file in pcd_list:
        file_name=file.split("/")[-1].split('.')[0]
        pcd=np.loadtxt(file)
        img = draw_point_cloud(pcd, zrot=90 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,
                               diameter=4)
        img=(img*255).astype(np.uint8)
        image_save_path=os.path.join(image_save_dir,file_name+".png")
        cv2.imwrite(image_save_path,img)
