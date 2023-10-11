import torch.nn as nn
import torch.nn.functional as func

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


model = FCN(num_classes=1)