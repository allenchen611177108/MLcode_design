import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

img_transform = transforms.Compose([transforms.ToTensor()])

mask_transform = transforms.Compose([transforms.ToTensor()])

class dataset(Dataset):
    def __init__(self, root, img_trans = None, mask_trans = None):
        self.dir_root = root
        self.img_path = os.path.join(self.dir_root, 'pic')
        self.mask_path = os.path.join(self.dir_root, 'mask')
        self.img_list = os.listdir(self.img_path)
        self.mask_list = os.listdir(self.mask_path)
        self.img_trans = img_trans
        self.mask_trans = mask_trans
    def __getitem__(self, index):
        img_name = self.img_list[index]
        file_path = os.path.join(self.img_path, img_name)
        img = Image.open(file_path).convert('RGB')
        if self.img_trans is not None:
            img = self.img_trans(img)

        mask_name = self.mask_list[index]
        maskfile_path = os.path.join(self.mask_path, mask_name)
        mask = Image.open(maskfile_path).convert('L')
        if self.mask_trans is not None:
            mask = self.mask_trans(mask)

        return img, mask
    def __len__(self):
        return len(self.img_list)
    
start = "C:/allen_env\deeplearning\metaDataset"
dir = []
for dirlv1 in (os.listdir(start)):
    node1 = os.path.join(start, dirlv1)
    for dirlv2 in (os.listdir(node1)):
        node2 = os.path.join(node1, dirlv2)
        dir.append(node2)


torch.set_printoptions(threshold=np.inf)
for site in dir:
    d = dataset(os.path.join(site, 'train'), img_trans=img_transform, mask_trans=mask_transform)
    img, mask = d[0]
    img = img.cuda()
    print(img)