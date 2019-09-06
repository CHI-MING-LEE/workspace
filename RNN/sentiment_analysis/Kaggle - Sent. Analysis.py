'''
LSTM Practice
@Date: 2018/07/15
@reference: https://blog.csdn.net/William_2015/article/details/72978387
'''

from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential, Input
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk
import collections
import numpy as np
import pandas as pd
import os

# Data preperation
"""
     在開始前，先對所用數據做個初步探索。
     特別地，我們需要知道數據中有多少個不同的單詞，每句話由多少個單詞組成。
"""

maxlen = 0  # max length of sentences
word_freqs = collections.Counter()  # ferq of words --> 以字典計數
num_recs = 0  # sample size

os.chdir("E:\\Users\\Ross\\Downloads\\Python\\Kaggle\\Sentiment")

# 算唯一單詞數量、最大句子長度
label = []
s = []

with open('train.txt', 'r+') as f:
    d = f.readlines()
    # print(max(list(map(len, x))))
    for line in d:
        sent_ = nltk.word_tokenize(line.lower())
        # print(sent_)
        label.append(sent_[0])  # 紀錄標籤
        sentence = sent_[1:]
        s.append(sentence)  # 紀錄句子

        if len(sentence) > maxlen:
            maxlen = len(sentence)
        for word in sentence:
            word_freqs[word] += 1
        num_recs += 1
print("max length: ", maxlen)  # 42
print('nb_words: ', len(word_freqs))  # 2324

# 從list/ dict 轉到 DF ===============================
# reference: http://pbpython.com/pandas-list-dict.html

# list到DF --> by col
"""
pd.DataFrame.from_items([("a", [1, 2, 3]), ("b", [4, 5, 6])])
結構: pd.DataFrame.from_items( list( tuple1, tuple2, ... ) )
tuple內: (colname, lsit of values [])
"""
# =====================================================

MAX_FEATURES = 2000  # 詞彙表大小設定為2000
MAX_SENTENCE_LENGTH = 40  # 最大句子長度

#  接下來建立兩個lookup tables，分別是word2index和index2word，用於單詞和數字轉換。
vocab_size = min(MAX_FEATURES, len(word_freqs))
word2index = {x[0]: i + 2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}  # 頻率最高前2000
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v: k for k, v in word2index.items()}

"""
下面就是根據lookup table把句子轉換成數字序列了，
並把長度統一到MAX_SENTENCE_LENGTH，不夠的填0 ，多出的截掉。
"""

X = np.empty(num_recs, dtype=list)  # 訓練資料句子數目
y = np.array(label, dtype='int')  # label

i = 0
for words in s:  # 把句子拉出來
    seqs = []
    for word in words:  # 把單字拉出來
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])  # 沒有的就用假單字補
    X[i] = seqs  # 第i句
    i += 1

print(len(X))
# Pads sequences to the same length.
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)  # 補齊為最大句子長度
"""
Notice: Sequences longer than num_timesteps are truncated 
        so that they fit the desired length.
"""

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

"""
 數據準備好後，就可以上模型了。這裡損失函數用binary_crossentropy，優化方法用adam。
 至於EMBEDDING_SIZE, HIDDEN_LAYER_SIZE,以及訓練時用到的BATCH_SIZE和NUM_EPOCHS這些超參數，
 就憑經驗多跑幾次調優了。
"""
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64

model_input = Input(shape=(40,))
model = Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH)(model_input)
model = LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2)(model)
model = Dense(1, activation='sigmoid')(model)

model_LSTM = Model(inputs=model_input, outputs=model)

model_LSTM.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# fit the model
BATCH_SIZE = 32
NUM_EPOCHS = 10

model_LSTM.fit(Xtrain,
               ytrain,
               batch_size=BATCH_SIZE,
               epochs=NUM_EPOCHS,
               validation_data=(Xtest, ytest))

# Prediction
"""
我們用已經訓練好的LSTM 去預測已經劃分好的測試集的數據，查看其效果。
選了5個句子的預測結果，並打印出了原句。
"""

score, acc = model_LSTM.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('Predict', 'real', 'sentence'))
for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1, 40)
    ylabel = ytest[idx]
    ypred = model_LSTM.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print('pred: {}, ground: {}, sentence: {}'.format(int(round(ypred)), int(ylabel), sent))

"""
我們可以自己輸入一些話，讓網絡預測我們的情感態度。
假如我們輸入I love reading. 和You are so boring. 兩句話，看看訓練好的網絡能否預測出正確的情感。
"""


# 一個句子
def interact_pred():
    INPUT_SENTENCES = input("請輸入一個句子:")

    words = nltk.word_tokenize(INPUT_SENTENCES.lower())
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index["UNK"])

    XX = sequence.pad_sequences(sequences=[seq], maxlen=MAX_SENTENCE_LENGTH) # list假設多個，所以這邊要放多層
    label_pred = int(round(model_LSTM.predict(XX)[0][0]))
    label2word = {1: 'Positive', 0: 'Negative'}
    print("Your sentence: '{}' belongs to {} emotion.".format(INPUT_SENTENCES, label2word[label_pred]))

interact_pred()


# 多個句子
INPUT_SENTENCES = ['I love reading.', 'You are so boring.']
XX = np.empty(len(INPUT_SENTENCES), dtype=list)
i = 0
for sentence in INPUT_SENTENCES:
    words = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i += 1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
labels = [int(round(x[0])) for x in model.predict(XX)]
label2word = {1: '积极', 0: '消极'}
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))
