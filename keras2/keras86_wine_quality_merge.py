# 실습
# 맹그러봐!!!
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#1. 데이터
dataset = pd.read_csv('../data/csv/winequality-white.csv', sep = ';').astype('float32')

x_data = dataset.iloc[:,:-1]
y_data = dataset.iloc[:,-1]
y_data[y_data<6] = 0
y_data[y_data==6] = 1
y_data[y_data>6] = 2
print(y_data.value_counts())

# print(np.unique(y_data)) # [3. 4. 5. 6. 7. 8. 9.]
# print(x_data) # [4898 rows x 11 columns]
# print(np.min(y_data)) # 3~9 사이로 측정

x = np.array(x_data)
y = np.array(y_data).reshape(-1, 1)

encoder = OneHotEncoder()
encoder.fit(y)
y = encoder.transform(y).toarray()

# print(x.shape, y.shape) (4898, 11) (4898, 7)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 42)
scale = RobustScaler()
scale.fit(x_train)

x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
x_val = scale.transform(x_val)

#2. 모델링
model = Sequential()
model.add(Dense(1024, activation= 'relu', input_shape = (11,)))
model.add(Dense(512, activation= 'relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(3, activation= 'softmax'))

#3. 컴파일 훈련
es = EarlyStopping(patience = 20)
lr = ReduceLROnPlateau(factor = 0.25, verbose = 1, patience = 10)
model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics = ['acc'])
model.fit(x_train, y_train, batch_size = 16, epochs = 1000, validation_data=(x_val, y_val), callbacks = [es, lr])

#4. 평가
loss = model.evaluate(x_test, y_test, batch_size= 16)
print('loss : ', loss[0])
print('acc : ', loss[1])

# Robust
# loss :  2.4030537605285645
# acc :  0.6428571343421936
