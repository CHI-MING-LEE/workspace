'''
@Date: 2018/7/16
@author: https://blog.csdn.net/aliceyangxi1987/article/details/73420583

同場加映: https://blog.csdn.net/yangyuwen_yang/article/details/82218100
'''

import os
import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

os.getcwd()
os.chdir("D:\\Users\\ROSS.LEE\\Desktop\\code\\Python\\Deep Learning\\RNN")

dataframe = pd.read_csv("international-airline-passengers.csv", usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values

# 將整數變為float
dataset = dataset.astype('float32')

plt.plot(dataset)
plt.show()


# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
# look_back就是預測下一步所需要的time steps數
# convert as array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):  # -1是因為從0開始
        a = dataset[i:(i + look_back), 0]  # t期:t+lb期 -> 不包含t+lb期
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])  # t+1期
    return np.array(dataX), np.array(dataY)


# fix random seed for reproducibility
np.random.seed(7)

# when activation function is sigmoid/ tahn --> need to scale the data, because LSTM is more sensitive now
# 設定67%是training，剩下的是testing

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split it into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# the data of X = t & Y = t + 1
look_back = 1
trainX, trainY = create_dataset(train, look_back)  # 因為look_back=1 => x是len為1的list
testX, testY = create_dataset(test, look_back)

"""
The structure of LSTM should be:
[samples, time steps, features]
"""

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# 建立LSTM模型：
# 输入层有 1 个input，隐藏层有 4 个神经元，输出层就是预测一个值，激活函数用 sigmoid，迭代 100 次，batch size 为 1
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back), return_sequences=False))  # (samples, time steps)
model.add(Dense(1))
model.add(LeakyReLU())
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
model.summary()
# make prediction
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 計算誤差之前要把值轉換回來
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# 计算 mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# 画出结果：蓝色为原数据，绿色为训练集的预测值，红色为测试集的预测值
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict  # t:t+1預測t+2

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
