import numpy as np
import os
import math
np.seterr(divide='ignore', invalid='ignore')
import random
from PIL import Image
import skimage
from skimage import transform
import matplotlib.pyplot as plt
from random import uniform
from skimage import io
import random
import pickle
import random
import sys
import cv2
import h5py
from PIL import Image
import matplotlib

import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.patches as patches
from pylab import *


gt_path = "./SYN/GT/"
gts = os.listdir(gt_path)
gts.sort()

haze_path = "./SYN/Haze/"
hazes = os.listdir(haze_path)
hazes.sort()

real_path = "./REAL/Haze/"
reals = os.listdir(real_path)
reals.sort()

data = zip(gts,hazes,reals)
data = np.array(list(data))


def random_crop_and_resize(image, size1=256,size2=320):

    #image = resize_image(image)

    h, w = image.shape[:2]

    y = np.random.randint(0, h-size1)
    x = np.random.randint(0, w-size2)

    image = image[y:y+size1, x:x+size2, :]

    return image



i = 0
for gt,haze,real in data:
   

    gt_image = np.float32(io.imread(gt_path + gt))/255.0
    gt_image = transform.resize(gt_image, (256, 320))
    #gt_image = random_crop_and_resize(gt_image)
    #gt_image = gt_image[:,:,:3]
    #print('gt.shape:', gt_image.shape)


    haze_image = np.float32(io.imread(haze_path + haze))/255.0
    haze_image = transform.resize(haze_image, (256, 320))
    #print('haze.shape:', haze_image.shape)

    real_image = np.float32(io.imread(real_path + real))/255.0
    real_image = transform.resize(real_image, (256, 320))
    #print('real_image.shape:', real_image.shape)

    
    f = h5py.File("./data/train/"+ str(i)+'.h5', "w")
    
    f.create_dataset('gt', data=gt_image)
    
    f.create_dataset('haze',data=haze_image)
    
    f.create_dataset('real', data=real_image)
    i=i+1
    print(i) 
     

    

    
 




