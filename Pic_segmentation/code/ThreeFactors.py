import os
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision import transforms
from PIL import Image
import numpy
import pickle
from torchmetrics.classification import BinaryJaccardIndex

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
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        data_name = self.imgs[idx]
        img_path = os.path.join(self.root, "img", data_name)
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


class SAFCN(nn.Module):
    def __init__(self, num_classes):
        super(SAFCN, self).__init__()

        self.factor_a = 1.
        self.factor_b = 1.
        self.factor_c = 1.

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

        score_fconn = self.factor_c * score_fconn
        score = func.upsample_bilinear(score_fconn, score_feat4.size()[2:])  # upsample_bilinear may be outdated
        score += self.factor_b * score_feat4
        score = func.upsample_bilinear(score, score_feat3.size()[2:])
        score += self.factor_a * score_feat3

        output = func.upsample_bilinear(score, x.size()[2:])  # 1,num_classes,256,256

        return output


validset = dataset('C:/allen_env/deeplearning/dataset/validation', transform=img_trans,
                   mask_transform=mask_trans)
vertifyloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=True, num_workers=0)

Path = 'C:/Users/arlun/PycharmProjects/pythonProject/result/model/FCN_Epoch300_20230609.pth'

test_net1 = SAFCN(2)
test_net1.load_state_dict(torch.load(Path))
test_net1.cuda()
test_net1.eval()

factor_list = numpy.array(
    [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

factor = []
IOU = []
try:
    for i in range(len(factor_list)):
        total_IOU = 0.0
        length = 0
        for j in range(len(factor_list)):
            for k in range(len(factor_list)):
                f_group = []
                a = factor_list[i]
                b = factor_list[j]
                c = factor_list[k]

                f_group.append(a)
                f_group.append(b)
                f_group.append(c)
                factor.append(f_group)

                test_net1.factor_a = a
                test_net1.factor_b = b
                test_net1.factor_c = c
                for index, data in enumerate(vertifyloader, 0):
                    img, mask_origin = data[0].cuda(), data[1].cuda()
                    predict1 = test_net1(img)

                    predict1 = predict1.cpu()

                    sigmoid = nn.Sigmoid()
                    predict1 = sigmoid(predict1)
                    threshold = torch.tensor([0.5])

                    mask = mask_origin[0][0].cpu()

                    # 實測後，[0][1] 最能表達理想結果
                    predict1_1 = (predict1[0][1] > threshold).float()
                    metric = BinaryJaccardIndex()
                    iou = metric(predict1_1, mask)
                    # print("iou: %s" %iou.item())
                    if torch.isnan(iou).any():
                        iou_current = 0.0
                    else:
                        iou_current = iou.item()
                    total_IOU = total_IOU + iou_current
                    # print("total_iou: %s" %total_IOU)
                    length += 1
                    torch.cuda.empty_cache()
                avg_iou = float(total_IOU / length)
                IOU.append(avg_iou)
                total_IOU = 0.0
                length = 0
except Exception as err:
    print(err)

max_index = IOU.index(max(IOU))
f_max = factor[max_index]

print(str(max(IOU)))
print(str(f_max))

file = open("result/exp0627.txt", "w")
file.write("all factor pair:\n")
pickle.dump(factor, file)
file.write('\r\n')
file.write("all factor pair:\n")
pickle.dump(IOU, file)
file.write('\r\n')
file.write("best factor pair:%s \n" %(f_max))
file.write("best IOU:%s" %(max(IOU)))
file.close()

torch.cuda.empty_cache()
