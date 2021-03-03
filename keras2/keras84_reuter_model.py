from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, LSTM, Embedding
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train),(x_test, y_test) = reuters.load_data( 
    num_words = 10000, test_split = 0.2
)

# 500개 를 맥스렌으로 잡겠다!!
maxlen = 500
x_train = pad_sequences(x_train, maxlen = maxlen, padding = 'pre', truncating= 'pre')
x_test = pad_sequences(x_test, maxlen = maxlen, padding = 'pre', truncating= 'pre')

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델
model = Sequential()
model.add(Embedding(10000, 100, input_length = maxlen)) # 단어를 100차원의 벡터로 변환해준다!
model.add(LSTM(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(46, activation= 'softmax'))

model.summary()

#3. 컴파일 훈련
lr = ReduceLROnPlateau(factor = 0.25, patience = 10)
es = EarlyStopping(patience = 20)
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])
model.fit(x_train,y_train, validation_data=(x_val, y_val), epochs = 1000, callbacks = [es, lr])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('정확도 : ', acc)
