import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# 1. data

x_train = np.array([[1,2,3,4,5,6,7,8,9,10,11,12],[5,7,9,11,13,15,17,19,21,23,25,27]])
y_train = np.array([[9,13,17,21,25,29,33,37,41,45,49,53]])
x = np.transpose(x_train)
y = np.transpose(y_train)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, shuffle=True)
x_test2 = ([[16,15,21],[35,5,45]])
x_test2 = np.transpose(x_test2)
y_test2 = ([[69,37,89]])
y_test2 = np.transpose(y_test2)


# 2. modelling

inputs = Input(shape = (2,))
dense1 = Dense(50)(inputs)
dense2 = Dense(50)(dense1)
dense3 = Dense(50)(dense2)
dense4 = Dense(50)(dense3)
dense5 = Dense(50)(dense4)
output = Dense(1)(dense5)
model = Model(inputs= inputs, outputs = output)
model.summary()

# 3. compile fit

model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs =100,validation_split = 0.2,batch_size =1)

# 4. evaluate predict

# loss, mae = model.evaluate(x_test,y_test,batch_size=1)
# print("loss : ", loss)
# print("mae : ", mae)
y_predict = model.predict(x_test2)
print(y_test2)
print(y_predict)
# print(y_predict)
from sklearn.metrics import r2_score
r2 = r2_score(y_test2,y_predict)
print("R2: ",r2)
