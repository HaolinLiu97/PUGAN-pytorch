import torch
import torch.nn as nn
import os,sys
sys.path.append('../')
from auction_match import auction_match
import pointnet2.pointnet2_utils as pn2_utils
import math
from knn_cuda import KNN

class Loss(nn.Module):
    def __init__(self,radius=1.0):
        super(Loss,self).__init__()
        self.radius=radius
        self.knn_uniform=KNN(k=2,transpose_mode=True)
        self.knn_repulsion=KNN(k=20,transpose_mode=True)
    def get_emd_loss(self,pred,gt,radius=1.0):
        '''
        pred and gt is B N 3
        '''
        idx, _ = auction_match(pred.contiguous(), gt.contiguous())
        #gather operation has to be B 3 N
        #print(gt.transpose(1,2).shape)
        matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
        matched_out = matched_out.transpose(1, 2).contiguous()
        dist2 = (pred - matched_out) ** 2
        dist2 = dist2.view(dist2.shape[0], -1)  # <-- ???
        dist2 = torch.mean(dist2, dim=1, keepdims=True)  # B,
        dist2 /= radius
        return torch.mean(dist2)
    def get_uniform_loss(self,pcd,percentage=[0.004,0.006,0.008,0.010,0.012],radius=1.0):
        B,N,C=pcd.shape[0],pcd.shape[1],pcd.shape[2]
        npoint=int(N*0.05)
        loss=0
        further_point_idx = pn2_utils.furthest_point_sample(pcd.contiguous(), npoint)
        new_xyz = pn2_utils.gather_operation(pcd.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
        for p in percentage:
            nsample=int(N*p)
            r=math.sqrt(p*radius)
            disk_area=math.pi*(radius**2)/N

            idx=pn2_utils.ball_query(r,nsample,pcd.contiguous(),new_xyz.permute(0,2,1).contiguous()) #b N nsample

            expect_len=math.sqrt(disk_area)

            grouped_pcd=pn2_utils.grouping_operation(pcd.permute(0,2,1).contiguous(),idx)#B C N nsample
            grouped_pcd=grouped_pcd.permute(0,2,3,1) #B N nsample C

            grouped_pcd=torch.cat(torch.unbind(grouped_pcd,dim=1),dim=0)#B*N nsample C

            dist,_=self.knn_uniform(grouped_pcd,grouped_pcd)
            #print(dist.shape)
            uniform_dist=dist[:,:,1:] #B*N nsample 1
            uniform_dist=torch.abs(uniform_dist+1e-8)
            uniform_dist=torch.mean(uniform_dist,dim=1)
            uniform_dist=(uniform_dist-expect_len)**2/(expect_len+1e-8)
            mean_loss=torch.mean(uniform_dist)
            mean_loss=mean_loss*math.pow(p*100,2)
            loss+=mean_loss
        return loss/len(percentage)
    def get_repulsion_loss(self,pcd,h=0.0005):
        dist,idx=self.knn_repulsion(pcd,pcd)#B N k

        dist=dist[:,:,1:5]**2 #top 4 cloest neighbors

        loss=torch.clamp(-dist+h,min=0)
        loss=torch.mean(loss)
        #print(loss)
        return loss
    def get_discriminator_loss(self,pred_fake,pred_real):
        real_loss=torch.mean((pred_real-1)**2)
        fake_loss=torch.mean(pred_fake**2)
        loss=real_loss+fake_loss
        return loss
    def get_generator_loss(self,pred_fake):
        fake_loss=torch.mean((pred_fake-1)**2)
        return fake_loss
    def get_discriminator_loss_single(self,pred,label=True):
        if label==True:
            loss=torch.mean((pred-1)**2)
            return loss
        else:
            loss=torch.mean((pred)**2)
            return loss
if __name__=="__main__":
    loss=Loss().cuda()
    point_cloud=torch.rand(4,4096,3).cuda()
    uniform_loss=loss.get_uniform_loss(point_cloud)
    repulsion_loss=loss.get_repulsion_loss(point_cloud)

