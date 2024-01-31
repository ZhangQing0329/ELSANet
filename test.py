
import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES']='1'
import cv2
import numpy as np
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import data
#from res_lx_2dai import resnet50
import time
from ELSANet import ELSANet

class Test(object):
    def __init__(self, Dataset, Network, path,dpath,epoch):
        ## dataset
        self.cfg    = Dataset.Config(datapath=dpath,listpath=path, snapshot='../ablation_pth/33/epoch'+str(epoch)+'.pth', mode='test')
    

        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.net.load_state_dict(torch.load(self.cfg.snapshot))
    def save(self,dataset):
        with torch.no_grad():
            time_t = 0.0

            for image, shape, name in self.loader:
                image = image.cuda().float()
                time_start = time.time()
                res, sal2_pred, sal3_pred, sal4_pred, sal5_pred, edge1_pred,edge2_pred, edge3_pred = self.net(image)
               
                torch.cuda.synchronize()
                time_end = time.time()
                time_t = time_t + time_end - time_start
               # res = res+sal2_pred
                
                save_path  ='../ablation_map/33_6_'+str(epoch)+'/'+dataset
                
                res = F.interpolate(res, shape, mode='bilinear', align_corners=True)
                #res1 = np.squeeze(res1.cpu().data.numpy())
                res = (torch.sigmoid(res[0,0])*255).cpu().numpy()

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # print(save_path+'/'+name[0]+'.png')
                cv2.imwrite(save_path+'/'+name[0]+'.png', res)

            fps = len(self.loader) / time_t
            print('FPS is %f' %(fps))



if __name__=='__main__':
    #testdata=['DUT-OMRON','DUTS-TE','ECSSD','HKU-IS','PASCAL-S']
    #testdata=['ECSSD','HKU-IS','PASCAL-S']
    testdata=['DUT-OMRON', 'DUTS-TE']

    for epoch in [65]:
        for dataset in testdata:
            for path in ['/home/linyi/Desktop/datasets/RGB_dataset/'+dataset+'/test.lst']:
                datapath='/home/linyi/Desktop/datasets/RGB_dataset/'+dataset+'/Imgs/'
                print(1)
                test = Test(data, ELSANet, path,datapath,epoch)
                test.save(dataset)
