import numpy as np
import torch
from Model_define_pytorch import Encoder, DatasetFolder
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random

# 计算 Pd（检测率）
def calculate_pd(tp, fn):
    return tp / (tp + fn)

# 计算 Pfa（虚警率）
def calculate_pfa(fp, tn):
    return fp / (fp + tn)


if __name__ == '__main__':
    # Parameters for training
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    use_single_gpu = True  # select whether using single gpu or multiple gpus
    # torch.manual_seed(42)
    batch_size = 256
    epochs = 100
    learning_rate = 1e-4
    num_workers = 8
    print_freq = 100  # print frequency (default: 60)

    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    random.seed(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # parameters for data

    # Model construction
    model = Encoder()
    if use_single_gpu:
        model = model.cuda()
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    import scipy.io as scio
   
    data_load_address = '../data'
    mat = np.load('../dataset/train-dataset/home_2023_10_30.npz')
    x_train = mat['data']  # shape=8000*126*128*2

    x_train = np.transpose(x_train.astype('float32'), [3,0,1,2])
    x_label = torch.tensor([int(item) for item in mat['ground_truth']])
    print(x_train.shape)
    print(x_label.shape)
    # Data loading

    x_train, x_test, y_train, y_test =train_test_split(x_train, x_label, test_size=0.2, random_state=2, stratify = x_label)
    
    train_dataset = DatasetFolder(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # mat = scio.loadmat(data_load_address+'/Htest.mat')
    # x_test = mat['H_test']  # shape=2000*126*128*2
    # x_test = np.transpose(x_test.astype('float32'), [0,3,1,2])
    # dataLoader for training
    test_dataset = DatasetFolder(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    best_loss = 1000
    best_score = 0
    best_pd = 0

    # nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='prelu')

    model_path = '../Modelsave'
    # model.encoder.load_state_dict(torch.load(model_path+'/encoder.pth.tar')['state_dict'])
    # model.decoder.load_state_dict(torch.load(model_path+'/decoder.pth.tar')['state_dict'])
    print("weight loaded")
    criterion = nn.CrossEntropyLoss().cuda()  #nn.MSELoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0)
    # schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.6)
    for epoch in range(epochs):
        # model training
        model.train()
        for i, (input, label) in enumerate(train_loader):
            # adjust learning rate
            input, label = input.cuda(), label.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, label)
            # compute gradient and do Adam step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss:.6f}\t'.format(
                    epoch, i, len(train_loader), loss=loss.item()))
        # model evaluating
        model.eval()
        total_loss = 0
        correct = 0  # 正确分类的样本数
        total = 0  # 总样本数
        tp = fn = fp = tn = 0
        Pd = 0  # 计算检测率（Probability of Detection）
        Pfa = 0  # 计算虚警率（Probability of False Alarm）
        occupied_time = 0  # 根据你的需求确定有人时长
        unoccupied_time = 0  # 根据你的需求确定无人时长
        
        with torch.no_grad():
            # for i, (input, label) in enumerate(test_loader):
            #     input, label = input.cuda(), label.cuda()
            #     output = model(input)
            #     total_loss += criterion(output, label).item() * label.size(0)
            # average_loss = total_loss / len(test_dataset)
            # print("Eval Loss: {:.6f}".format(average_loss))
            # if average_loss < best_loss:
            #     torch.save({'state_dict': model.state_dict(), }, model_path+'/encoder.pth.tar')
            #     print("Model saved")
            #     best_loss = average_loss

            # # 这段代码计算准确率
            # for i, (input, label) in enumerate(test_loader):
            #     input, label = input.cuda(), label.cuda()
            #     output = model(input)
                
            #     _, predicted = torch.max(output, 1)  # 获取模型的预测标签
            #     total += label.size(0)  # 更新总样本数
            #     correct += (predicted == label).sum().item()  # 更新正确分类的样本数

            # accuracy = 100 * correct / total  # 计算准确率
            # print("Accuracy: {:.2f}%".format(accuracy))
            # if accuracy > best_accuracy:
            #     torch.save({'state_dict': model.state_dict(), }, model_path+'/encoder.pth.tar')
            #     print("Model saved")
            #     best_accuracy = accuracy

            # 这段代码计算分数
            for i, (input, label) in enumerate(test_loader):
                input, label = input.cuda(), label.cuda()
                output = model(input)
                
                _, predicted = torch.max(output, 1)  # 获取模型的预测标签
                total += label.size(0)  # 更新总样本数
                tp += (torch.logical_and(label > 0, predicted > 0)).sum().item() # 真阳性样本数，label有人，predict有人
                fn += (torch.logical_and(label > 0, predicted == 0)).sum().item()# 假阴性样本数，label有人，predict无人
                fp += (torch.logical_and(label == 0, predicted > 0)).sum().item()# 假阳性样本数，label无人，predict有人
                tn += (torch.logical_and(label == 0, predicted == 0)).sum().item()# 真阴性样本数，label无人，predict无人
                correct += (torch.logical_and(predicted == label, label > 0)).sum().item()  # 更新人数识别准确样本数
                occupied_time += (label != 0).sum().item()   # 有人样本数，label有人
                unoccupied_time += (label == 0).sum().item()  # 无人样本数，label无人

            # 计算 Pd 和 Pfa
            pd = calculate_pd(tp, fn)
            pfa = calculate_pfa(fp, tn)
            # 打印结果
            print("Pd (Probability of Detection): {:.2f}".format(pd))
            print("Pfa (Probability of False Alarm): {:.2f}".format(pfa))
            # print("Acc : {:.2f}".format(correct / (tp+fp)))
            if tp+fp == 0:
                score = 0.8 *(occupied_time / total * pd + unoccupied_time / total * (1-pfa))
            else:
                score = 0.8 *(occupied_time / total * pd + unoccupied_time / total * (1-pfa)) + 0.2 * (correct / (tp+fn))
            print("Score: {:.6f}".format(score))
            if score > best_score and pfa <= 0.05:
                torch.save({'state_dict': model.state_dict(), }, model_path+'/encoder.pth.tar')
                print("Model saved")
                best_score = score

        schedule.step()
