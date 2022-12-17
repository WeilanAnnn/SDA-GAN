from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torchvision.transforms as transforms
#import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import cv2
from networks import *
from skimage import measure
import time
import torchvision.utils as vutils
from testdata_process.misc import *
from testdata_process.image_dataset import TestDataset
import torchvision.models as models
import h5py
import torch.nn.functional as F
import numpy as np
from ptflops import get_model_complexity_info
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()


parser.add_argument('--test_dataset', type=str, default="./test/", help=' path of images')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--mode', type=str, default='REAL', help='REAL: Your data is real, SYN: Your data is synthetic')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
opt = parser.parse_args()

if parser.parse_args().mode is 'REAL':
    parser.add_argument('--encoder', default="./model/For_real/encoder_epoch_real.pth", help="path to netG (to continue training)")
    parser.add_argument('--decoder', default="./model/For_real/decoder_epoch_real.pth", help="path to netG (to continue training)")

elif parser.parse_args().mode is 'SYN':
    parser.add_argument('--encoder', default="./model/For_syn/encoder_epoch_syn.pth", help="path to netG (to continue training)")
    parser.add_argument('--decoder', default="./model/For_syn/decoder_epoch_syn.pth", help="path to netG (to continue training)")

else :
    print(' #####AttributeError! Please choose your data type, \'REAL\' or \'SYN\'##### ')

opt = parser.parse_args()
    
# get dataloader
test_dataset = TestDataset(opt.test_dataset)
Dataloader_test = DataLoader(dataset=test_dataset, batch_size=opt.batchSize, shuffle=True)

opt.workers=1


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
print(device_ids)
if torch.cuda.is_available():
    print( torch.cuda.is_available())
    device = torch.device('cuda')



# --- Define the network --- #


encoder = Semi_Encoder()
encoder_state_dict = torch.load(opt.encoder)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in encoder_state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
encoder.load_state_dict(new_state_dict)
encoder = nn.DataParallel(encoder).to(device)

decoder = Semi_Decoder()
decoder_state_dict = torch.load(opt.decoder)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in decoder_state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
decoder.load_state_dict(new_state_dict)
decoder = nn.DataParallel(decoder).to(device)

encoder.train()
decoder.train()


# --- Calculate all trainable parameters in network --- #
print('#####################################################################################################')
print('    Total params_Encoder: %.2fM' % (sum(p.numel() for p in encoder.parameters())/1000000.0))
print('    Total params_Decoder: %.2fM' % (sum(p.numel() for p in decoder.parameters())/1000000.0))
print('#####################################################################################################')






directory='./result/DHQ/'

if not os.path.exists(directory):
  os.makedirs(directory)
t = 0

if not os.path.exists(directory):
  os.makedirs(directory)



with torch.no_grad():
    for i, data in enumerate(Dataloader_test, 0):
        test_haze, name = data
        test_haze = test_haze.to(device)
        start = time.time()
        haze_latent = encoder(test_haze)
        dehazed = decoder(haze_latent)  
        end = time.time()              
        dehazed1 = dehazed.squeeze(0)
        print(end-start)
        index = name[0]
        vutils.save_image(dehazed1, directory + index + '_new.jpg', normalize=True, scale_each=False)
        t0 = end-start
        t = t + t0
                
        print(i)

print('Finish')
