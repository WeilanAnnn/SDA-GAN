import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import h5py
import glob
import scipy.ndimage
IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
  images = []
  if not os.path.isdir(dir):
    raise Exception('Check dataroot')
  for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
      if is_image_file(fname):
        path = os.path.join(dir, fname)
        item = path
        images.append(item)
  return images

def default_loader(path):
  return Image.open(path).convert('RGB')

class data_read(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None):
    # imgs = make_dataset(root)
    # if len(imgs) == 0:
    #   raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
    #              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root
    # self.imgs = imgs
    self.transform = transform
    self.loader = loader

    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):
    # index = np.random.randint(1,self.__len__())
    # index = np.random.randint(self.__len__(), size=1)[0]

    # path = self.imgs[index]
    # img = self.loader(path)
    #img = img.resize((w, h), Image.BILINEAR)



    file_name=self.root+str(index)+'.h5'
    f=h5py.File(file_name,'r')

    haze=f['haze'][:]
    #haze=np.resize(haze, (256, 256, 3))

    
    gt=f['gt'][:]
    #gt = np.resize(gt, (256, 256,3))
    
    real = f['real'][:]
    #real = np.resize(real, (256, 256, 3))



    haze=np.swapaxes(haze,0,2)
    
    gt=np.swapaxes(gt,0,2)
    
    real = np.swapaxes(real,0,2)
    



    haze=np.swapaxes(haze,1,2)
    
    gt=np.swapaxes(gt,1,2)
    
    real = np.swapaxes(real,1,2)


    return haze,gt,real

  def __len__(self):
    train_list=glob.glob(self.root+'/*h5')
    # print len(train_list)
    return len(train_list)

    # return len(self.imgs)
