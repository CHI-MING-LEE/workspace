'''
@ Date of review: 2018/07/17
@ Author: SK.Lin


'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, LeakyReLU, BatchNormalization
from keras.models import Sequential
from keras.callbacks import TensorBoard
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import os

os.getcwd()
os.chdir('D:\\Users\\ROSS.LEE\\Desktop\\code\\Python\\Deep Learning\\RNN')
os.listdir()

SHIFT = 8
BATCH = 128
df = pd.read_csv('week.csv', names=['Date', 'Counts'])  # (144, 2)

df.head(15)

# adfuller(df.Counts.values, autolag='AIC')[1] # pvlaue = 0.02954060071056521
# acorr_ljungbox(df.Counts.values, lags=1)[1] # pvalue = 8.04656879e-09

"""
因為有Lead time為8周的關係，所以x取8周，從第1周開始；y從第15周開始

"""
X = df.Counts[:-8]  # 136 --> 最後8周不考慮(沒有更多資料了for y)
Y = df.Counts[15:]  # 129 --> 預測第15周為第一筆
X = np.asarray(X).reshape((-1, 1))
Y = np.asarray(Y).reshape((-1, 1))

# 訓練資料為80%的資料
X_train = X[:int(X.shape[0] * 0.8)]  # 0:108
X_test = X[int(X.shape[0] * 0.8):]  # 108:136
# y已經經過shift，所以每8筆x對到1筆y
# -> 108 - 8 + 1 = 101 對到101筆y
Y_train = Y[:(108 - 7)]  # 101
# -> 136 - 108 - 8 + 1 = 21
Y_test = Y[-(28 - 7):]  # 21 --> 預測7周後的話這裡的確是最後21周

# 進行scale的動作
X_scale = MinMaxScaler(feature_range=(0, 100))
x_train = X_scale.fit_transform(X_train)
Y_scale = MinMaxScaler(feature_range=(0, 100))
y_train = Y_scale.fit_transform(Y_train)


def get_batch(data_x, data_y, batch_size, time_step):
    while True:
        # [samples, time steps, features]
        x_batch = np.zeros(shape=[batch_size, time_step, 1], dtype=np.float32)
        y_batch = np.zeros(shape=[batch_size, 1], dtype=np.float32)

        # 隨機從資料中生成batch_size筆的x_batch和y_batch
        for i in range(batch_size):
            index = np.random.randint(len(data_y))

            x_batch[i] = data_x[index: index + SHIFT].reshape([SHIFT, 1])
            y_batch[i] = data_y[index].reshape([1])  # 已經進行過SHIFT

        yield (x_batch, y_batch)


generator = get_batch(x_train, y_train, BATCH, SHIFT) # 一個batch 8天(8筆資料)
# 可以看下一圈
m, n = next(generator)

# m.shape
model = Sequential()
model.add(LSTM(input_shape=(None, 1), units=30, return_sequences=False, activation='relu'))
model.add(Dense(units=1))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.summary()
model.compile(optimizer='RMSProp', loss='mse', metrics=['mape'])
# callback = TensorBoard(batch_size=BATCH)
history = model.fit_generator(generator=generator, steps_per_epoch=128, epochs=150)  # , callbacks=[callback])


def make_pred_data(input_data, step_time):
    pred = np.zeros(shape=(input_data.shape[0] - step_time, step_time, 1))
    # 因為要取step_time周所以要減掉 28 - 8 = 20
    for i in range(input_data.shape[0] - step_time):
        pred[i] = input_data[i:i + step_time]
    return pred


test_scale = MinMaxScaler(feature_range=(0, 100))
x_test = test_scale.fit_transform(X_test)
x_2_test = make_pred_data(x_test, SHIFT)
pred = model.predict(x_2_test)
pred_inverse = test_scale.inverse_transform(pred)
# pred_inverse = X_scale.inverse_transform(pred)
# y_test_inverse = Y_scale.inverse_transform((y_test))
# prediction = np.append([0]*16, pred_inverse)
# df['prediction'] = prediction
# df.plot()
import matplotlib.pyplot as plt

plt.plot(Y_test, 'r-')
plt.plot(pred_inverse)
# plt.legend()
plt.title('TPCC8131')
plt.show()
df.Counts[-20:].plot()


# 生成器可以迭代

def yield_func():
    a = 0
    for i in range(5):
        a += i
        yield a


for i in yield_func():
    print(i)

# 要丟進變數才能不每次都初始化
generator = yield_func()
g = next(generator)
