import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.feats1 = nn.Sequential(self.features[0:5])
        self.feats2 = nn.Sequential(self.features[5:10])
        self.feats3 = nn.Sequential(self.features[10:17])
        self.feats4 = nn.Sequential(self.features[17:24])
        self.feats5 = nn.Sequential(self.features[24:31])

        self.fconn = nn.Sequential(
            nn.Conv2d(512, 2560, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(2560, 2560, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)
        self.score_fconn = nn.Conv2d(2560, num_classes, 1)

    def forward(self, x):
        # Size of input=1,num_classes,256,256
        feats1 = self.feats1(x)  # 1,128,64,64
        feats2 = self.feats2(feats1)  # 1,256,32,32
        feats3 = self.feats3(feats2)  # 1,512,16,16
        feats4 = self.feats4(feats3)  # 1,512,8,8
        feats5 = self.feats5(feats4)  # 1,512,8,8
        fconn = self.fconn(feats5)  # 1,2560,8,8

        score_feat3 = self.score_feat3(feats3)  # 1,num_classes,32,32
        score_feat4 = self.score_feat4(feats4)  # 1,num_classes,16,16
        score_fconn = self.score_fconn(fconn)  # 1,num_classes,8,8

        score = func.upsample_bilinear(score_fconn, score_feat4.size()[2:])  # upsample_bilinear may be outdated
        score += score_feat4
        score = func.upsample_bilinear(score, score_feat3.size()[2:])
        score += score_feat3

        output = func.upsample_bilinear(score, x.size()[2:])  # 1,num_classes,256,256

        return output

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
        img = torch.unsqueeze(img, dim=0) # neural network`s input is not a image, is a image in the "batch"
        mask_name = self.mask_list[index]
        maskfile_path = os.path.join(self.mask_path, mask_name)
        mask = Image.open(maskfile_path).convert('L')
        if self.mask_trans is not None:
            mask = self.mask_trans(mask)
        return img, mask
    def __len__(self):
        return len(self.img_list)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

inner_lr = 0.001
lr = 0.05
model = FCN(num_classes=2)
model.cuda()
model_copy = FCN(num_classes=2)
model_copy.load_state_dict(model.state_dict())
model_copy.cuda()

criterion = nn.CrossEntropyLoss()
meta_optimizer = optim.SGD(model_copy.parameters(), lr=inner_lr, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

metalr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, mode='min', factor=0.5, patience=4, min_lr=0.000001, cooldown=1)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, min_lr=0.000001)

start = "C:/allen_env\deeplearning\metaDataset"
dir = []
for dirlv1 in (os.listdir(start)):
    node1 = os.path.join(start, dirlv1)
    for dirlv2 in (os.listdir(node1)):
        node2 = os.path.join(node1, dirlv2)
        dir.append(node2)

model_copy.train()
model.train()
torch.autograd.set_detect_anomaly(True)
for i in range(10):
    print("Meta Epoch: %d \n" %i)
    meta_optimizer.zero_grad()
    meta_loss = 0
    for site in dir:
        # every tasks
        d = dataset(os.path.join(site, 'train'), img_trans=img_transform, mask_trans=mask_transform)
        d_test = dataset(os.path.join(site, 'test'), img_trans=img_transform, mask_trans=mask_transform)
        seq_list = random.sample(range(0, d.__len__()), 5)
        for idx in seq_list:
            meta_optimizer.zero_grad()
            img, mask = d[idx]
            output = model_copy(img.cuda())
            loss = criterion(output, mask.long().cuda())
            loss.cpu()
            print("loss: %f \n" %loss)
            loss.backward()
            meta_optimizer.step()
            metalr_scheduler.step(loss)
            torch.cuda.empty_cache()
        idx_test = random.randint(0, d_test.__len__() - 1)
        img_test, mask_test = d_test[idx_test]
        output_test = model_copy(img_test.cuda())
        task_loss = criterion(output_test, mask_test.long().cuda())
        task_loss.cpu()
        print("task_loss: %f \n" %task_loss)
        task_loss = task_loss.detach()
        meta_loss = meta_loss + task_loss
        torch.cuda.empty_cache()
    optimizer.zero_grad()
    meta_loss = meta_loss / len(dir)
    print("meta_loss: %f \n" %meta_loss)
    meta_loss = Variable(meta_loss, requires_grad = True)
    meta_loss.backward()
    optimizer.step()
    lr_scheduler.step(meta_loss)
    torch.cuda.empty_cache()

torch.save({
            'model_state_dict':model.state_dict(),
            }, 'C:/allen_env/deeplearning/result/model/FCN_Model_meta_0.pth')