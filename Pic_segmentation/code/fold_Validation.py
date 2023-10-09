import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torchvision import transforms
from PIL import Image
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

batch_size = 1
bias = 1e-5

mask_trans = transforms.Compose(
    [transforms.ToTensor()]
)
img_trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, mask_transform=None):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "pic"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        data_name = self.imgs[idx]
        img_path = os.path.join(self.root, "pic", data_name)
        mask_path = os.path.join(self.root, "mask", data_name)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)

class rand_dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, mask_transform=None):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "pic"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.imgs) - 1)
        data_name = self.imgs[idx]
        img_path = os.path.join(self.root, "pic", data_name)
        mask_path = os.path.join(self.root, "mask", data_name)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)

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

trainset = dataset('C:/allen_env/deeplearning/7f/fold_7/train_set', transform=img_trans,
                   mask_transform=mask_trans)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

# randset = rand_dataset('C:/allen_env/deeplearning/7f/fold_3/train_set', transform=img_trans,
#                    mask_transform=mask_trans)
# randloader = torch.utils.data.DataLoader(randset, batch_size=1, shuffle=True, num_workers=0)

# validset = dataset('C:/allen_env/deeplearning/7f/fold_3/test_set', transform=img_trans,
#                    mask_transform=mask_trans)
# vertifyloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=True, num_workers=0)


model_path = "C:/allen_env\deeplearning/result\model\FCN_Model_meta_0.pth"
model = FCN(2)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

target_epoch = 300

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=80, min_lr=0.000001, cooldown=20)

model.train()

IOU = []

try:
    for epoch in range(target_epoch):
        print(epoch)
        for i, data in enumerate(trainloader, 0):
            img, mask = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, mask[0].long()).cpu()
            loss.cpu().backward()
            optimizer.step()
            torch.cuda.empty_cache()
            if(epoch+1 > 100):
                lr_scheduler.step(loss)
        if (epoch+1 % 160 == 0):
            torch.save({
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()
            }, 'C:/allen_env/deeplearning/result/checkpoint/meta_FCN0_cp_7fold_7.pth')
    torch.save(model, 'C:/allen_env/deeplearning/result/model/Meta_FCN0_Epoch300_7fold_7.pth')
except Exception as err:
    print(err)

print("=========================================================== \n")

# model.eval()
# try:
#     total_IOU = 0.0
#     length = 300
#     for img, mask in itertools.islice(randloader, length):
#         img, mask = img.cuda(), mask.cuda()
#         predict = model(img)

#         predict = predict.cpu()

#         sigmoid = nn.Sigmoid()
#         predict1 = sigmoid(predict)
#         threshold = torch.tensor([0.5])

#         mask = mask[0][0].cpu()

#         # 實測後，[0][1] 最能表達理想結果
#         predict1_1 = (predict[0][1] > threshold).float() * 1
#         metric = BinaryJaccardIndex()
#         metric.update(predict1_1, mask)
#         iou = metric.compute()
#         # print("iou: %s. \n" %iou.item())
#         # print("============================================\n")
#         if torch.isnan(iou).any():
#             iou_current = 0.0
#         else:
#             iou_current = iou.item()
#         total_IOU = total_IOU + iou_current
#         torch.cuda.empty_cache()
#     avg_iou = float(total_IOU / length)
#     print("average_Train_iou = %f.\n" %avg_iou)
# except Exception as err:
#     print(err)

# try:
#     total_IOU = 0.0
#     length = 300
#     for img, mask in itertools.islice(vertifyloader, length):
#         img, mask = img.cuda(), mask.cuda()
#         predict = model(img)

#         predict = predict.cpu()

#         sigmoid = nn.Sigmoid()
#         predict1 = sigmoid(predict)
#         threshold = torch.tensor([0.5])

#         mask = mask[0][0].cpu()

#         # 實測後，[0][1] 最能表達理想結果
#         predict1_1 = (predict[0][1] > threshold).float() * 1
#         metric = BinaryJaccardIndex()
#         metric.update(predict1_1, mask)
#         iou = metric.compute()
#         # print("iou: %s. \n" %iou.item())
#         # print("============================================\n")
#         if torch.isnan(iou).any():
#             iou_current = 0.0
#         else:
#             iou_current = iou.item()
#         total_IOU = total_IOU + iou_current
#         torch.cuda.empty_cache()
#     avg_iou = float(total_IOU / length)
#     print("average_Valid_iou = %f." %avg_iou)
# except Exception as err:
#     print(err)

torch.cuda.empty_cache()