import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class TestDataset(Dataset):
    def __init__(self, root,unaligned= False, mode='val'):
        self.transform = transforms.Compose(
         [#transforms.Resize((384,512), Image.BICUBIC),
			   #transforms.RandomCrop(opt.size),
			   #transforms.RandomHorizontalFlip(),
			   transforms.ToTensor()])
			   #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root) + '/*.*'))
        
    def __getitem__(self, index):
        img = self.files_A[index]
        
        item_A = self.transform(Image.open(img))
        name = img.split("/")[-1]
        name, _ = name.split('.')
            

        return item_A,name

    def __len__(self):
        return len(self.files_A)        
        