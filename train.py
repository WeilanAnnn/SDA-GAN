from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from traindata_process import pix2pix
from torch.utils.data import DataLoader
import os
import cv2
import time
import torchvision.utils as vutils
from traindata_process import pix2pix, data_read, misc
from traindata_process.misc import *
from utils import get_local_time
from networks import *
from utils import Timer
import numpy as np
import matplotlib.pyplot as plt
import sys
from torch.nn import functional as F
import shutil
from skimage.measure import compare_mse
from utils import weights_init, get_model_list, get_scheduler
from losses import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
                    default='pix2pix', help='')
                    
'''
 How to generate your own Training/Validation dataset:
 1.Prepare your paired synthetic data and real data, then put them in three separate folders, like 'haze','gt','real'
 2.Run 'generate_trainingdata.py' to generate your h5py files and save them.
 3.'--dataroot': Put your folder path
                 
'''                    
parser.add_argument('--dataroot', required=False,
                    default="./train/", help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
                    default="./val/", help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=60, help='epoch to reaching at learning rate of 0')
parser.add_argument('--encoder', default="", help="path to net (to continue training)")
parser.add_argument('--decoder', default="", help="path to net (to continue training)")
parser.add_argument('--map', default="", help="path to net (to continue training)")
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in net')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay for adam')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='checkpoint', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=50,
                    help='interval for evauating(generating) images from valDataroot')
parser.add_argument('--Diters', type=int, default=1, help='Number of iterations of D')
parser.add_argument('--Giters', type=int, default=1, help='Number of iterations of G.')
parser.add_argument('--Gan_lambda', type=int, default=0.1, help='Number of iterations of D')

opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
#opt.manualSeed = random.randint(1, 10000)
opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
opt.workers = 1

# get dataloader
dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True,
                       seed=opt.manualSeed)



valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False,
                          seed=opt.manualSeed)



# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
# print(device_ids)
if torch.cuda.is_available():
    # print( torch.cuda.is_available())
    device = torch.device('cuda')

# built training network

encoder = Semi_Encoder()
encoder.apply(weights_init('kaiming'))
encoder = nn.DataParallel(encoder).to(device)
if opt.encoder != '':
   encoder.load_state_dict(torch.load(opt.encoder))

decoder = Semi_Decoder()
decoder.apply(weights_init('kaiming'))
decoder = nn.DataParallel(decoder).to(device)
if opt.decoder != '':
   decoder.load_state_dict(torch.load(opt.decoder))


enc_params = list(encoder.parameters()) 
dec_params = list(decoder.parameters()) 


dis1 = NLayerDiscriminator_dark()
dis1.apply(weights_init('kaiming'))
dis1 = nn.DataParallel(dis1).to(device)

dis2 = NLayerDiscriminator()
dis2.apply(weights_init('kaiming'))
dis2 = nn.DataParallel(dis2).to(device)

enc_opt = torch.optim.Adam([p for p in enc_params if p.requires_grad],lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.0001)
dec_opt = torch.optim.Adam([p for p in dec_params if p.requires_grad],lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.0001)

dis_opt1 = torch.optim.Adam(dis1.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.0001)
dis_opt2 = torch.optim.Adam(dis2.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.0001)


# get randomly sampled validation images and save it
valDataloader_ = getLoader(opt.dataset,
                          opt.valDataroot,
                          len(valDataloader),
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False,
                          seed=opt.manualSeed)
val_iter = iter(valDataloader_)
data_val = val_iter.next()

val_haze, val_gt, val_real = data_val
val_haze, val_gt, val_real = val_haze.to(device), val_gt.to(device), val_real.to(device)

vutils.save_image(val_gt, '%s/syn_gt.png' % opt.exp, normalize=True)
vutils.save_image(val_haze, '%s/syn_haze.png' % opt.exp, normalize=True)
vutils.save_image(val_real, '%s/real_haze.png' % opt.exp, normalize=True)

# --- Calculate all trainable parameters in network --- #
print('#####################################################################################################')
print('    Total params_encoder: %.2fM' % (sum(p.numel() for p in encoder.parameters())/1000000.0))
print('    Total params_decoder: %.2fM' % (sum(p.numel() for p in decoder.parameters())/1000000.0))
print('#####################################################################################################')



kl_loss = KLDivergence()
kl_loss.to(device)

criterionCAE = nn.L1Loss()
criterionCAE.to(device)
criterionBCE = nn.BCELoss()
criterionBCE.to(device)



max_iter = 100000
iteration =0
G_loss=0
for epoch in range(opt.annealStart, opt.num_epochs):

    start_time = time.time()
    batch_id = 0
    if epoch > opt.annealStart:
        adjust_learning_rate(enc_opt, opt.lr, epoch, None, opt.annealEvery)
        adjust_learning_rate(dec_opt, opt.lr, epoch, None, opt.annealEvery)
        adjust_learning_rate(dis_opt1, opt.lr, epoch, None, opt.annealEvery)
        adjust_learning_rate(dis_opt2, opt.lr, epoch, None, opt.annealEvery)


    for i, data in enumerate(dataloader, 0):

        haze, gt, real = data
        haze = haze.type(torch.FloatTensor)
        haze = haze.to(device)
        gt = gt.type(torch.FloatTensor)
        gt = gt.to(device)
        real = real.type(torch.FloatTensor)
        real = real.to(device)


        for p in dis2.parameters():
            p.requires_grad = True
        for p in dis1.parameters():
            p.requires_grad = True
        #start to update D
        for t in range(opt.Diters):
            #--- Zero the parameter gradients --- #
            dis2.train(),dis1.train()
            dis_opt2.zero_grad()
            dis_opt1.zero_grad()
            
            real_latent = encoder(real.detach())
            
            haze_latent = encoder(haze.detach())
            
            pred_fake_0 = dis2(real_latent)
            pred_real_0 = dis2(haze_latent)

            errD_real_0 = criterionBCE(pred_real_0, torch.ones_like(pred_real_0))
            errD_fake_0 = criterionBCE(pred_fake_0, torch.zeros_like(pred_fake_0))
            
            real_dehaze = decoder(real_latent)
            haze_dehaze = decoder(haze_latent)
            dark_realdehaze = dark(real_dehaze)
            dark_gt = dark(gt)
            pred_fake_1 = dis1(dark_realdehaze)
            pred_real_1 = dis1(dark_gt)            

            errD_real_1 = criterionBCE(pred_real_1, torch.ones_like(pred_real_1))
            errD_fake_1 = criterionBCE(pred_fake_1, torch.zeros_like(pred_fake_1))
                        
            errD_0 = errD_real_0 + errD_fake_0
            errD_1 = errD_real_1 + errD_fake_1
            
            errD = opt.Gan_lambda* errD_0 + 0.5 *opt.Gan_lambda * errD_1

            errD.backward()
            dis_opt2.step()
            dis_opt1.step()

        #prevent computing gradients of weights in Discriminator
        for p in dis2.parameters():
            p.requires_grad = False
        for p in dis1.parameters():
            p.requires_grad = False


        # start to update G
        for t in range(opt.Giters):
            enc_opt.zero_grad()
            dec_opt.zero_grad()

            encoder.train()
            decoder.train()
            

            haze_latent = encoder(haze)
            haze_dehaze = decoder(haze_latent)

            real_latent = encoder(real)
            real_dehaze = decoder(real_latent)

            haze_dehaze_l1 = criterionCAE(gt, haze_dehaze)
            

            hazereal_kl = 10*kl_loss(real_latent, haze_latent)


            pred_real_0 = dis2(haze_latent)
            
            dark_realdehaze = dark(real_dehaze)
            pred_real_1 = dis1(dark_realdehaze)
            
            errG_0 = criterionBCE(pred_real_0, torch.ones_like(pred_real_0))
            errG_1 = criterionBCE(pred_real_1, torch.ones_like(pred_real_1))
            
            errG = opt.Gan_lambda *errG_0 + 0.5 *opt.Gan_lambda * errG_1
            
            loss_G = haze_dehaze_l1 + hazereal_kl + errG 

            loss_G.backward()
            enc_opt.step()
            dec_opt.step()

            G_loss += loss_G.item()
            epoch_loss = G_loss / (iteration+1)
            iteration += 1

        if iteration % 50 == 0:

            print('#########################################################')
            print('[%d/%d][%d/%d]' % (epoch, opt.num_epochs, batch_id, len(dataloader)), 'epoch_loss:', epoch_loss)
            print('#########################################################')

        if iteration % 5 == 0:

            print("<{}> Iteration: %08d/%08d".format(get_local_time()) % (iteration + 1, max_iter))
            print('[%d/%d][%d/%d]' % (epoch, opt.num_epochs, i, len(dataloader)))
            print(' errD_0: %04f  errG_0: %04f errD_1: %04f  errG_1: %04f  haze_dehaze_l1: %04f  hazereal_kl: %04f ' \
            %(errD_0.item(),errG_0.item(),errD_1.item(),errG_1.item(),haze_dehaze_l1.item(),hazereal_kl.item()))
            
      
        

          
    if epoch % 1 == 0:
        val_batch_output_syn = torch.FloatTensor(len(valDataloader), val_haze.size(1), val_haze.size(2),val_haze.size(3)).fill_(0)
        val_batch_output_syndehaze = torch.FloatTensor(len(valDataloader), val_haze.size(1), val_haze.size(2),val_haze.size(3)).fill_(0)
        val_batch_output_gt = torch.FloatTensor(len(valDataloader), val_haze.size(1), val_haze.size(2),val_haze.size(3)).fill_(0)
        val_batch_output_realdehaze = torch.FloatTensor(len(valDataloader), val_haze.size(1), val_haze.size(2),val_haze.size(3)).fill_(0)
        with torch.no_grad():


              for i, val in enumerate(valDataloader, 0):
                  val_haze, val_gt, val_real = val
                  val_haze = val_haze.type(torch.FloatTensor)
                  val_haze = val_haze.to(device)
                  haze_latent = encoder(val_haze)
                  haze_dehaze = decoder(haze_latent)
                  haze_dehaze1 = haze_dehaze.squeeze(0)
                  val_batch_output_syndehaze[i, :, :, :].copy_(haze_dehaze1)

                  val_real = val_real.type(torch.FloatTensor)
                  val_real = val_real.to(device)
                  real_latent = encoder(val_real)
                  real_dehaze = decoder(real_latent)
                  real_dehaze1 = real_dehaze.squeeze(0)
                  val_batch_output_realdehaze[i, :, :, :].copy_(real_dehaze1)
                
              vutils.save_image(val_batch_output_syndehaze, '%s/syn_dehazed_epoch_%03d_iter%06d.png' % \
                                (opt.exp, epoch, batch_id), normalize=True, scale_each=False)         
              vutils.save_image(val_batch_output_realdehaze, '%s/real_dehazed_epoch_%03d_iter%06d.png' % \
                                (opt.exp, epoch, batch_id), normalize=True, scale_each=False)       
              
          

    # # --- Save the network parameters ---
    if epoch % 1 == 0:
        torch.save(encoder.state_dict(), '%s/encoder_epoch_%d.pth' % (opt.exp, epoch))
        torch.save(decoder.state_dict(), '%s/decoder_epoch_%d.pth' % (opt.exp, epoch))
    
