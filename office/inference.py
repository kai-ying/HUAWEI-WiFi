#!/usr/bin/env python3
import numpy as np
from Model_define_pytorch import Encoder
import torch
import os

# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
batch_size = 64
# load test data
mat = np.load('../dataset/test-dataset/home_2023_10_31.npz')
x_test = mat['data']  # shape=2000*126*128*2

x_test = torch.tensor(np.transpose(x_test.astype('float32'), [3,0,1,2]))
# load model
model = Encoder().cuda()
model_path = '../Modelsave'
model.load_state_dict(torch.load(model_path+'/encoder.pth.tar')['state_dict'])
print("weight loaded")

#dataLoader for test

# test
model.eval()
res = []
with torch.no_grad():
    for i, input in enumerate(x_test):
        # convert numpy to Tensor
        input = input.cuda()
        output = model(input.unsqueeze(0))
        res.append(int(torch.argmax(output).cpu().numpy()))
        # output = output.cpu().numpy()
# 最后保存到文件就好了
print(res)



# truth = mat['ground_truth'].astype(int)
# # print(truth[1000:2000])
# # # print(len(truth))
# print(res[0:1000] - truth[0:1000])