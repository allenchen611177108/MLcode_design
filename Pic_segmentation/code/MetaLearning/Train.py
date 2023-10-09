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

"""Learner Class

Design Target：Use Factory Pattern to fulfill a learner can use for learning and meta-learning
"""
class Learner():
    def __init__(self, model, hyper_param = {'lr':0.01, 'Epoch':50}, optimize_group = None):
        self.Epoch = hyper_param['Epoch']
        self.Model = model

        if (optimize_group != None):
            self.criterion = optimize_group['criterion']
            self.optimizer = optimize_group['optimizer']
            self.lrSchedular = optimize_group['lr_schedular']
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.Model.parameters(), momentum=0.9)
            self.lrSchedular = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.optimizer.lr = hyper_param['lr']
    def Train(self):
        pass

class SegLearner(Learner):
    def __init__(self, model, hyper_param, optimize_group):
        super().__init__(model, hyper_param, optimize_group)

    def Train(self, root, target, transform = None):
        if(transform == None):
            mask_trans = transforms.Compose([transforms.ToTensor()])
            img_trans = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485,0.426,0.406], std=[0.229,0.224,0.225])])
        else:
            mask_trans = transforms['maskTransform']
            img_trans = transforms['imageTransform']
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        self.Model.cuda()
        Dataset = dataset(root, img_trans=img_trans, mask_trans=mask_trans)
        trainloader = torch.utils.data.DataLoader(Dataset, batch_size=1, shuffle=True, num_workers=0)
        self.Model.train()
        for epoch in range(self.Epoch):
            print(epoch)
            for i, data in enumerate(trainloader, 0):
                img, mask = data[0].cuda(), data[1].cuda()
                optimizer.zero_grad()
                output = self.Model(img)
                loss = criterion(output, mask).cpu()
                loss.cpu().backward()
                optimizer.step()
                torch.cuda.empty_cache()
                lr_scheduler.step()
        torch.save(target)

class MAMLLearner(Learner):
    def __init__(self, model, hyper_param, optimize_group, meta_optimize=None, meta_param = {'lr':0.05, 'task_epoch':5}):
        super().__init__(model, hyper_param, optimize_group)
        self.meta_model = self.Model
        self.Task_Epoch = meta_param['task_epoch']
        if(meta_optimize == None):
            self.meta_criterion = nn.CrossEntropyLoss()
            self.meta_optimizer = optim.SGD(self.meta_model.parameters(), momentum=0.9)
            self.metalr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optimizer, mode='min', factor=0.5,
                                                                          patience=4, min_lr=0.000001, cooldown=1)
        else:
            self.meta_criterion = meta_optimize['criterion']
            self.meta_optimizer = meta_optimize['optimizer']
            self.metalr_scheduler = meta_optimize['lr_schedular']
        self.meta_optimizer.lr = meta_param['lr']

    def Train(self, root, target, transform = None):
        if(transform == None):
            mask_trans = transforms.Compose([transforms.ToTensor()])
            img_trans = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485,0.426,0.406], std=[0.229,0.224,0.225])])
        else:
            mask_trans = transforms['maskTransform']
            img_trans = transforms['imageTransform']
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        for i in range(self.Epoch):
            print("Meta Epoch: %d \n" % i)
            self.meta_optimizer.zero_grad()
            meta_loss = 0
            for site in os.listdir(root):
                # every tasks
                query_set = dataset(os.path.join(root, site, 'test'), img_trans=img_trans, mask_trans=mask_trans)
                query_loader = torch.utils.data.DataLoader(query_set, batch_size=1, shuffle=True)
                self.task_learn(os.path.join(root, site, 'train'))
                query_img, query_mask = next(iter(query_loader))
                query_img = query_img[0].squeeze()
                query_mask = query_mask[0]
                query_output = self.meta_model(query_img.cuda())
                task_loss = self.criterion(query_output, query_mask.long().cuda())
                task_loss.cpu()
                print("task_loss: %f \n" % task_loss)
                task_loss = task_loss.detach()
                meta_loss = meta_loss + task_loss
                torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            meta_loss = meta_loss / len(os.listdir(root))
            print("meta_loss: %f \n" % meta_loss)
            meta_loss = Variable(meta_loss, requires_grad=True)
            meta_loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(meta_loss)
            torch.cuda.empty_cache()
        torch.save({
            'model_state_dict': self.Model.state_dict(),
        }, target)

    def task_learn(self,path):
        support_set = dataset(path, img_trans=self.img_trans, mask_trans=self.mask_trans)
        support_loader = torch.utils.data.DataLoader(support_set, batch_size=self.Task_Epoch, shuffle=True)
        for img, mask in iter(support_loader):
            self.meta_optimizer.zero_grad()
            img = img.cuda()
            mask = mask.cuda()
            output = self.meta_model(img)
            loss = self.meta_criterion(output, mask.long().cuda())
            loss.cpu()
            print("loss: %f \n" % loss)
            loss.backward()
            self.meta_optimizer.step()
            self.metalr_scheduler.step(loss)
            torch.cuda.empty_cache()

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