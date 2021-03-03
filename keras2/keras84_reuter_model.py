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

vocab = 1000
#1. 데이터
(x_train, y_train),(x_test, y_test) = reuters.load_data( 
    num_words = vocab, test_split = 0.2
)

# 500개 를 맥스렌으로 잡겠다!!
maxlen = 200
x_train = pad_sequences(x_train, maxlen = maxlen, padding = 'pre')
x_test = pad_sequences(x_test, maxlen = maxlen, padding = 'pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
model = Sequential()
model.add(Embedding(vocab, 200, input_length = maxlen)) # 단어를 400차원의 벡터로 변환해준다!
model.add(LSTM(128))
model.add(Dense(54, activation= 'relu'))
model.add(Dense(46, activation= 'softmax'))

model.summary()

#3. 컴파일 훈련
lr = ReduceLROnPlateau(factor = 0.25, patience = 4)
es = EarlyStopping(patience = 8)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train,y_train, validation_split = 0.2, epochs = 1000 , callbacks = [es, lr], batch_size = 32)

#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print('정확도 : ', acc)

# 정확도 :  0.7373107671737671 # rmsprop
# 정확도 :  0.7560107111930847 # adam
# 고마워요 옵티박!