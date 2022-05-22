#!/usr/bin/env python
# coding: utf-8

# In[127]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[128]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[115]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[116]:


# 读取数据
import pandas as pd
import numpy as np
data = pd.read_csv("city_huang_all.csv")
data = np.array(data)
print(data)


# 对数据进行处理，提取武汉城市的每日新冠确诊

# In[117]:


pt = np.unique(data[:,2])
pt1 = np.arange(len(pt))
for i in range(len(pt)):
    data[data==pt[i]]=pt1[i]
pp = data[data[:,2]==136]
data_new = pp[:,[9]]
print(data_new)


# 生成可处理的数据格式[batch_size,time_steps,input_size]分别为（-1，6），即用6天的数据为一组，其中前五天数据为输入，第六天数据为输出

# In[118]:


num = 5
data_new_new = []
pst = []
for j in range(1):
    for t in range(len(data_new)-num-1):
        for z in range(num+1):
            pst.append(data_new[t+z,0])
        data_new_new.append(pst)
        pst = []  
print(data_new_new)


# In[119]:


#加载飞桨、Numpy和相关类库
import paddle
from paddle.nn import Linear
from paddle.nn import LSTM, MSELoss
import paddle.nn.functional as F
import numpy as np
import os
import random
import matplotlib.pyplot as plt


# 对数据进行归一化和拆分输入输出和训练集与测试集

# In[120]:


def load_data(data_new):
    # 从文件导入数据
    data = data_new
    data = np.array(data)
    data = data.astype('float32')
    np.random.shuffle(data)
    # 将原数据集拆分成训练集和测试集
    # 这里使用90%的数据做训练，10%的数据做测试
    ratio = 0.9
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0),                                  training_data.sum(axis=0) / training_data.shape[0]
    
    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    # 对数据进行归一化处理
    for i in range(6):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    training_data = np.reshape(training_data,(len(training_data),6,1))
    test_data = np.reshape(test_data,(len(test_data),6,1))
    return training_data, test_data


# 定义训练网络，这里使用的是一层LSTM加上一层全连接层，注意：LSTM层的输出包含很多元素，要对输出进行转化得到实际想要的输出

# In[121]:


class Regressor(paddle.nn.Layer):

    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Regressor, self).__init__()
        
        self.fc1 = LSTM(1, 32, 2)
        self.fc2 = Linear(32, 1)
    # 网络的前向计算
    def forward(self, inputs):
        x1, _ = self.fc1(inputs)
        x2= self.fc2(x1[:,-1,:])
        return x2


# 进行训练前的数据加载和优化算法的定义

# In[122]:


# 声明定义好的线性回归模型
model = Regressor()
# 开启模型训练模式
# paddle.set_device('gpu:0')
model.train()
# 加载数据
training_data, test_data = load_data(data_new_new)
# 定义优化算法，使用随机梯度下降SGD
# 学习率设置为0.01
opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
print(training_data)


# 进行训练

# In[123]:


EPOCH_NUM = 2000   # 设置循环次数
epoch_id = 0  # 记录已经循环的次数
loss_load = []
# 定义循环
for i in range(EPOCH_NUM):
    x = training_data[:, :-1, :] # 获得训练数据
    y = training_data[:, -1:, :] # 获得训练标签（真实确诊人数）
    y = np.reshape(y,(-1,1))
    # 将numpy数据转为tensor形式
    house_features = paddle.to_tensor(x)
    prices = paddle.to_tensor(y)
    
    # 前向计算
    predicts = model(house_features)
    # 计算损失
    loss = F.square_error_cost(predicts, label=prices)
    avg_loss = paddle.mean(loss)
    opt.minimize(avg_loss)
    epoch_id = epoch_id + 1
    if epoch_id%5==0:
        loss_load.append(avg_loss.numpy()[0])
        print("epoch: {}, loss is: {}".format(epoch_id, avg_loss.numpy()))
    
    # 反向传播
    avg_loss.backward()
    # 最小化loss,更新参数
    opt.step()
    # 清除梯度
    opt.clear_grad()


# In[124]:


loss_load = np.array(loss_load)
plot_x = np.arange(len(loss_load))
plt.plot(plot_x, loss_load, 'b')
plt.show()


# 保存模型

# In[125]:


# 保存模型参数，文件名为LR_model.pdparams
paddle.save(model.state_dict(), 'LR_model.pdparams')
print("模型保存成功，模型参数保存在LR_model.pdparams中")


# In[126]:


# 参数为保存模型参数的文件地址
model_dict = paddle.load('LR_model.pdparams')
model.load_dict(model_dict)
model.eval()

# 参数为数据集的文件地址
one_data = test_data[:, :-1] # 获得当前批次训练数据
label = test_data[:, -1:] # 获得当前批次训练标签（真实房价）
label = label*(max_values[-1] - min_values[-1])+avg_values[-1] #反归一化
# 将数据转为动态图的variable格式 
one_data = paddle.to_tensor(one_data)
predict = model(one_data)
predicts = predict.numpy()
predicts = predicts*(max_values[-1] - min_values[-1])+avg_values[-1]
label = np.reshape(label,(-1,1))

predicts = np.reshape(predicts,(-1,1))
plot_x = np.arange(len(predicts))
plt.plot(plot_x, predicts, 'b')
plt.plot(plot_x, label, 'r')
plt.show()
# 对结果做反归一化处理
# 对label数据做反归一化处理

print("Inference result is {}, the corresponding label is {}".format(predicts, label))


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
