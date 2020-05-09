import h5py
import torch.utils.data as data
import os, sys
sys.path.append("../")
import numpy as np
import utils.data_util as utils
from torchvision import transforms

class PUNET_Dataset_Whole(data.Dataset):
    def __init__(self, data_dir='../MC_5k',n_input=1024):
        super().__init__()
        self.raw_input_points=5000
        self.n_input=1024

        file_list = os.listdir(data_dir)
        self.names = [x.split('.')[0] for x in file_list]
        self.sample_path = [os.path.join(data_dir, x) for x in file_list]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        random_index=np.random.choice(np.linspace(0,self.raw_input_points,self.raw_input_points,endpoint=False),self.n_input).astype(np.int)
        points = np.loadtxt(self.sample_path[index])

        #centroid=np.mean(points[:,0:3],axis=0)
        #dist=np.linalg.norm(points[:,0:3]-centroid,axis=1)
        #furthest_dist=np.max(dist)

        #reduced_point=points[random_index][:,0:3]

        #normalized_points=(reduced_point-centroid)/furthest_dist

        return points#normalized_points,furthest_dist,centroid

class PUNET_Dataset(data.Dataset):
    def __init__(self, h5_file_path='../Patches_noHole_and_collected.h5',split_dir='./train_list.txt',
                 skip_rate=1, npoint=1024, use_random=True, use_norm=True,isTrain=True):
        super().__init__()

        self.isTrain=isTrain

        self.npoint = npoint
        self.use_random = use_random
        self.use_norm = use_norm

        h5_file = h5py.File(h5_file_path)
        self.gt = h5_file['poisson_4096'][:]  # [:] h5_obj => nparray
        self.input = h5_file['poisson_4096'][:] if use_random \
            else h5_file['montecarlo_1024'][:]
        assert len(self.input) == len(self.gt), 'invalid data'
        self.data_npoint = self.input.shape[1]

        centroid = np.mean(self.gt[..., :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((self.gt[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        self.radius = furthest_distance[:, 0]  # not very sure?

        if use_norm:
            self.radius = np.ones(shape=(len(self.input)))
            self.gt[..., :3] -= centroid
            self.gt[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
            self.input[..., :3] -= centroid
            self.input[..., :3] /= np.expand_dims(furthest_distance, axis=-1)

        self.split_dir = split_dir
        self.__load_split_file()

    def __load_split_file(self):
        index=np.loadtxt(self.split_dir)
        index=index.astype(np.int)
        print(index)
        self.input=self.input[index,:]
        self.gt=self.gt[index,:]
        self.radius=self.radius[index]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        input_data = self.input[index]
        gt_data = self.gt[index]
        radius_data = np.array([self.radius[index]])

        sample_idx = utils.nonuniform_sampling(self.data_npoint, sample_num=self.npoint)
        input_data = input_data[sample_idx, :]

        if not self.isTrain:
            return input_data, gt_data, radius_data

        if self.use_norm:
            # for data aug
            input_data, gt_data = utils.rotate_point_cloud_and_gt(input_data, gt_data)
            input_data, gt_data, scale = utils.random_scale_point_cloud_and_gt(input_data, gt_data,
                                                                               scale_low=0.9, scale_high=1.1)
            input_data, gt_data = utils.shift_point_cloud_and_gt(input_data, gt_data, shift_range=0.1)
            radius_data = radius_data * scale

            # for input aug
            #if np.random.rand() > 0.5:
            #    input_data = utils.jitter_perturbation_point_cloud(input_data, sigma=0.025, clip=0.05)
            #if np.random.rand() > 0.5:
            #    input_data = utils.rotate_perturbation_point_cloud(input_data, angle_sigma=0.03, angle_clip=0.09)
        else:
            raise NotImplementedError

        return input_data, gt_data, radius_data

if __name__=="__main__":
    dataset=PUNET_Dataset()
    #(input_data,gt_data,radius_data)=dataset.__getitem__(0)
    #print(input_data.shape,gt_data.shape,radius_data.shape)
    #dataset=PUNET_Dataset_Whole(data_dir="../MC_5k",n_input=1024)
    #points=dataset.__getitem__(0)
    #print(points.shape)