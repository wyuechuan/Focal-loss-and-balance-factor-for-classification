#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 11:19
# @Author  : Wang Yuechuan
# @FileName: _1_SatelliteFaultDiagnosis.py
# @Software: PyCharm
# @function: 

import time

since = time.time()

import numpy as np
import pandas as pd
import copy
import time
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import random
import os


# Split data set and preprocess
def load_dataset(dataInit):
    (numData, numFeat) = dataInit.shape
    data = dataInit.iloc[:, list(range(numFeat - 1))]
    label = dataInit.iloc[:, [numFeat - 1]]
    # 读入数据拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.4, random_state=0, stratify=label)
    X_test, X_ver, y_test, y_ver = train_test_split(X_test, y_test, test_size=0.5, random_state=0, stratify=y_test)


    # train_test_split 返回的数据格式和输入的数据格式一直(DataFrame,ndarray等)
    # 预处理
    X_test = X_test.values
    X_train = X_train.values
    y_train = y_train.values
    y_test = y_test.values
    X_ver = X_ver.values
    y_ver = y_ver.values
    scalerM = MinMaxScaler()
    scalerM.fit(X_train)
    X_train = scalerM.transform(X_train)  # 返回值是ndarray类型
    X_test = scalerM.transform(X_test)
    X_ver = scalerM.transform(X_ver)

    x_test = torch.from_numpy(X_test).float()
    x_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    X_ver = torch.from_numpy(X_ver).float()
    y_ver = torch.from_numpy(y_ver).float()

    return x_train, y_train, x_test, y_test, X_ver, y_ver


# Calculate the proportion of equality
def get_accuracy(y_pred, y_target):
    # Input the category
    n_correct = torch.eq(y_pred, y_target).sum().item()
    accuracy = n_correct / len(y_pred) * 100
    return accuracy


def get_class_perf(cm_perf):
    # 输入混淆矩阵，并从混淆矩阵中进行相关评价指标的计算
    np.set_printoptions(precision=4)  # 设置np打印精度
    precision = np.zeros(len(cm_perf))
    recall = np.zeros(len(cm_perf))
    acc = 0
    total = np.sum(cm_perf)
    for i, ele in enumerate(cm_perf):
        # return the index and item
        precision[i] = ele[i] / max(np.sum(cm_perf[:, i]), 1e-5)  # 防止为zero
        recall[i] = ele[i] / max(np.sum(ele), 1e-5)
        acc += ele[i]
    acc = np.array([acc / total])
    f1_score = 2 * precision * recall / np.clip(precision + recall, 1e-5, 2)
    print("The Confusion Matrix is")
    print(cm_perf)
    print("The  accuracy is", end=": ")
    print(acc)
    print("The precision is", end=": ")
    print(precision)
    print("The   recall  is", end=": ")
    print(recall)
    print("The  f1 score is", end=": ")
    print(f1_score)

    return acc, precision, recall, f1_score


def class_loss(y_pred, y_train, gamma=2.):
    p = torch.sigmoid(y_pred)
    loss = - (1 - p) ** gamma * torch.log(p) * y_train \
           - p ** gamma * torch.log(1 - p) * (1 - y_train) / len(y_train)
    loss_class = torch.zeros((2))
    loss_class[0] = (loss * y_train).sum()
    loss_class[1] = (loss * (1 - y_train)).sum()

    cv_loss_class = torch.std(loss_class) / torch.mean(loss_class)
    return loss_class, cv_loss_class


# 定义loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=4.5, alpha=0.05):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if gamma==0:
            self.x = 0
        else:
            self.x = 1
    def forward(self, preY, tureY):
        preY = torch.sigmoid(preY)
        lossList = ((- (1-self.alpha) ** self.x) *(1 - preY) ** self.gamma * torch.log(preY) * tureY \
                         - (self.alpha ** self.x) * preY ** self.gamma * torch.log(1 - preY) * (1 - tureY)) / len(tureY)
        loss = torch.sum(lossList)
        # 统计loss的分布特征
        loss_class = torch.zeros((2))
        loss_class[0] = (lossList.data * tureY).sum()
        loss_class[1] = (lossList.data * (1 - tureY)).sum()
        # cv_loss_class = loss_class[0] / loss_class[1]
        cv_loss_class = torch.std(loss_class)/torch.mean(loss_class)
        #  用来调试nan
        if torch.isnan(loss):
            print("The loss has been nan!!!")
            print(torch.sum(self.alpha * (1 - preY) ** self.gamma * torch.log(preY) * tureY))
            print(torch.sum(preY ** self.gamma * torch.log(1 - preY) * (1 - tureY)))
            print(torch.sum(torch.log(1 - preY)))
            print(torch.sum(preY))
            print(torch.min(preY))
            print(torch.min(1 - preY))
            print(torch.min(torch.log(preY)))
        return loss, cv_loss_class


# Multilayer Perception
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):  # 特征值数量、隐藏层节点数量([]),类别数量
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0], bias=True)
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[0])
        self.fc3 = nn.Linear(hidden_size[0], hidden_size[1], bias=True)
        self.final = nn.Linear(hidden_size[1], num_classes, bias=True)  # output layer

    def forward(self, x_in):  # 网络的输出未经过softmax运算
        # 定义前向之后，反向自动实现
        x = torch.relu(self.fc1(x_in))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        y_pred = self.final(x)

        return y_pred


def train_model(num_epochs, dataInit, gamma_set=4.5, alaph =0.05):
    # Load Dataset and Return the Tensor
    x_train, y_train, x_test, y_test, X_ver, y_ver = load_dataset(dataInit)
    category_t = 0.5
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Model configuration
    [m_train, n_train] = x_train.shape
    input_size = n_train
    hidden_size = [60, 30]
    # Train configuration
    learning_rate = 0.003  #0.003
    step_size = 200
    model = MLP(input_size=input_size,
                hidden_size=hidden_size,
                num_classes=1)

    # initNetParams(model)
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device).reshape((-1, 1)).float()  # 在loss中必须使用float
    x_test = x_test.to(device)
    y_test = y_test.to(device).reshape((-1, 1)).float()
    X_ver = X_ver.to(device)
    y_ver = y_ver.to(device).reshape((-1, 1)).float()

    # Optimization(Object and way of optimization)
    loss_fl = FocalLoss(gamma=gamma_set, alpha=alaph)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())  # 用于保存训练参数
    best_acc = 0.0  # 存储最好的测试精度
    all_loss = torch.zeros(num_epochs)
    all_loss_test = torch.zeros(num_epochs)
    cv_loss = torch.zeros(num_epochs)
    accList = torch.zeros(num_epochs)
    global y_pred  # 循环中的变量要在循环外进行引用
    # Training
    for t in range(num_epochs):
        # Forward pass
        y_pred = model(x_train)  # 不经过sigmoid
        # Accuracy
        predictions = (torch.sigmoid(y_pred) > category_t).long()
        accuracy = get_accuracy(y_pred=predictions, y_target=y_train)  # 计算正确的比例
        accList[t] = accuracy

        # Loss
        loss, cv_loss_class = loss_fl(y_pred, y_train)
        cv_loss[t] = cv_loss_class
        all_loss[t] = loss.data

        # Zero all gradients
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # Learning Rate update
        scheduler.step()
        # Test Dataset
        y_test_pre = torch.sigmoid(model(x_test))  # , apply_sigmoid=True
        pred_test = (y_test_pre) > category_t
        test_acc = get_accuracy(y_pred=pred_test, y_target=y_test)
        # 测试集LOSS
        loss_test, cv_loss_class = loss_fl(y_test_pre, y_test)
        all_loss_test[t] = loss_test.data

        # deep copy the model  保存测试精度最大的一个
        if test_acc > best_acc:
            # print("最好的测试精度更新：{0}, 训练精度为：{1}".format(t, accuracy))
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())  # copy.deepcopy表示一种复制，即保存测试集测试最好的一组模型
        if t % 100 == 0:
            print(
                "epoch: {0:4d} | loss: {1:.8f} | loss_veri: {4:.8f} | Train accuracy: {2:.4f}% | Veri accuracy: {3:.4f}%"
                .format(t, loss, accuracy, test_acc, loss_test))
    # End training

    # Save the model
    torch.save(best_model_wts, "model_ANN_Focal_loss.pt")  # best_model_wts是前面所保存下来的最好的模型参数in test set（字典参数）

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Predictions
    best_model_wts = torch.load("model_ANN_Focal_loss.pt")
    model.load_state_dict(best_model_wts)

    pred_train = (torch.sigmoid(model(x_train)) > category_t)
    pred_test = (torch.sigmoid(model(x_test)) > category_t)
    pred_ver = (torch.sigmoid(model(X_ver)) > category_t)
    # Train and test accuracies
    train_acc = get_accuracy(y_pred=pred_train, y_target=y_train)
    test_acc = get_accuracy(y_pred=pred_test, y_target=y_test)
    var_acc = get_accuracy(y_pred=pred_ver, y_target=y_ver)
    print("train acc: {0:.1f}%, test acc: {1:.1f}, ver acc: {2:.1f}%".format(
        train_acc, test_acc, var_acc))

    # Evaluate result in train set_by wyc
    y_train = y_train.cpu().numpy()
    predictions = pred_train.cpu().numpy()
    cm_perf_train = confusion_matrix(y_train, predictions)
    number_class_train = torch.sum(torch.from_numpy(cm_perf_train).to(device),
                                   dim=1)  # 能放到GPU上的都放到GPU上，但是后续进行其他操作可能需要换回来
    acc, precision, recall, f1_score = get_class_perf(cm_perf_train)
    model_perf_train = [acc, precision, recall, f1_score]

    # evaluate result in test set
    y_true = y_test.cpu().numpy()
    y_pred = pred_test.cpu().numpy()
    cm_perf_test = confusion_matrix(y_true, y_pred)  # in test set
    number_class = torch.sum(torch.from_numpy(cm_perf_test).to(device), dim=1)
    acc, precision, recall, f1_score = get_class_perf(cm_perf_test)
    model_perf = [acc, precision, recall, f1_score]

    cv_loss = cv_loss.cpu().numpy()
    all_loss = all_loss.numpy()
    accList = accList.numpy()

    return best_model_wts, accList, model_perf, all_loss, cv_loss, \
           number_class, model_perf_train, number_class_train, all_loss_test, var_acc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 简易网格搜索
def gridIteration():
    setup_seed(20)
    gammaList = [4.5]
    alphaList = [0.05]
    numGamma = len(gammaList)
    numAlpha = len(alphaList)
    gammaBest = 100
    alphaBest = 100
    accBest = 0
    for i in range(numGamma):
        for j in range(numAlpha):
            setup_seed(20)         # 这个很关键
            print("/*******************************/")
            print("gamma = {0:f}".format(gammaList[i]))
            print("alpha = {0:f}".format(alphaList[j]))
            dataInit = pd.read_csv("dataUnBalancedQian1.csv", index_col=0)
            print("开始训练......")
            best_model_wts, accList, model_perf_fl, fl_loss, cv_loss, number_class, model_perf_fl_train, \
            number_class_train, loss_test_fl, var_acc \
                = train_model(num_epochs=1001, dataInit=dataInit,
                              gamma_set=gammaList[i],alaph=alphaList[j])

            if model_perf_fl[3][1] > accBest:
                accBest = model_perf_fl[3][1]
                gammaBest = gammaList[i]
                alphaBest = alphaList[j]
            print("gammaBest={}".format(gammaBest))
            print("alphaBest={}".format(alphaBest))
            print("accBest={}".format(accBest))


if __name__ == '__main__':
    gridIteration()


time_elapsed = time.time() - since
print('program complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))