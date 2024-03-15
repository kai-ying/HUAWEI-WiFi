import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')


# Silu直接调用nn.SiLU，下面的改成F.silu

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1, padding=(2, 1), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=(2, 1), stride=(1,2)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=(2, 1), stride=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=(2, 1), stride=2),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        # self.net2 = nn.Sequential(
        #     nn.Conv2d(4, 16, kernel_size=3, padding=(2, 2), dilation=2),
        #     nn.BatchNorm2d(16),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, padding=(2, 2), dilation=2),
        #     nn.BatchNorm2d(16),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, padding=(2, 2), dilation=2),
        #     nn.BatchNorm2d(16),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 4, kernel_size=3, padding=(2, 2), dilation=2),
        #     nn.BatchNorm2d(4),
        #     nn.PReLU(),
        # )

        # self.transformer_dis = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=128*2, nhead=8), num_layers=1)

        # self.fc = nn.Linear(128*11*5, 256)

        self.fc = nn.Linear(128*11*16, 256)  # 修改全连接层输入维度
        self.fc2 = nn.Linear(256, 4)
        

    def forward(self, x):

        out = self.net(x)
        # print(out.shape)
        # out = self.net2(x) + x

        # out = torch.cat((out_real, out_imag), dim=1)


        # out_dis_seq = out.permute([0, 2, 1, 3])
        # out_angle_seq = out.permute([0, 3, 1, 2])
        # out_dis = self.transformer_dis(out_dis_seq.reshape([out_dis_seq.size(0), out_dis_seq.size(1), -1]))
        # out_angle = self.transformer_angle(out_angle_seq.reshape([out_angle_seq.size(0), out_angle_seq.size(1), -1]))

        # out = torch.cat((out_dis.reshape([out_dis.size(0), 1, -1]), \
        #     out_angle.reshape([out_angle.size(0), 1, -1])), dim=1)

        # out = out.view(-1, 128*11*5)
        # out = out.view(-1, 1056000)
        out = out.view(-1, 128*11*16)  # 修改视图大小以适应新的输入维度
        out = F.relu(self.fc(out))
        out = F.relu(self.fc2(out))
        return out



# dataLoader
class DatasetFolder(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]
    


# 下面是MLP-mixer的实现
class MIX_BLOCK(nn.Module):
    def __init__(self, d, res):
        super().__init__()
        self.lm = nn.LayerNorm(res)

        self.mlp1 = nn.Sequential(
            nn.Linear(d,256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256,d),
            nn.Dropout(0.1)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(res,2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048,res),
            nn.Dropout(0.1)
        )
    def forward(self,x):
        x = self.lm(x)
        out = torch.transpose(x,1,2)
        out = torch.transpose(self.mlp1(out),1,2)
        x = x + out
        x = self.lm(x)
        x = x + self.mlp2(x)
        return x



class MLP_MIXER(nn.Module):
    def __init__(self, inchannels, patch_size, dim, mix_num, image_size, num_classes):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.out_num = int(image_size // patch_size)**2
        self.block = MIX_BLOCK
        self.to_patch = nn.Conv2d(inchannels, dim, kernel_size = patch_size, stride = patch_size)

        self.mix_blocks = nn.ModuleList([])
        for i in range(mix_num):
            self.mix_blocks.append(MIX_BLOCK(self.out_num,dim))

        self.avgpool = nn.AvgPool1d(2)

        self.head = nn.Linear(int(dim/2)*self.out_num, num_classes)

    def forward(self,x):
        x = self.to_patch(x)
        x = x.view(x.shape[0],x.shape[1],-1)
        x = torch.transpose(x,1,2)
        for mix_block in self.mix_blocks:
            x = mix_block(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        x = self.head(x)
        return x


if __name__ == "__main__":
    # input = torch.ones([1,3,224,224])
    # model = MLP_MIXER(3, 16, 128, 12, 224, 4)
    # out = model(input)
    # print(out.shape)
    # exit(0)
    model = Encoder()
    # x = torch.randn(150, 4, 32, 64)
    x = torch.randn(150, 4, 32, 248) #    修改输入维度
    y = model(x)
    print(y.shape)

