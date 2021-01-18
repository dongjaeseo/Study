import numpy as np
import pandas as pd

# 제출용!!
x_samsung_train = np.load('./samsung/samsung_final_samsung.npy', allow_pickle = True)[0]
x_samsung_test = np.load('./samsung/samsung_final_samsung.npy', allow_pickle = True)[1]
x_samsung_val = np.load('./samsung/samsung_final_samsung.npy', allow_pickle = True)[2]
x_samsung_pred = np.load('./samsung/samsung_final_samsung.npy', allow_pickle = True)[9]

x_kodex_train = np.load('./samsung/samsung_final_kodex.npy', allow_pickle = True)[0]
x_kodex_test = np.load('./samsung/samsung_final_kodex.npy', allow_pickle = True)[1]
x_kodex_val = np.load('./samsung/samsung_final_kodex.npy', allow_pickle = True)[2]
x_kodex_pred = np.load('./samsung/samsung_final_kodex.npy', allow_pickle = True)[9]

y1_train = np.load('./samsung/samsung_final_samsung.npy', allow_pickle = True)[3]
y1_test = np.load('./samsung/samsung_final_samsung.npy', allow_pickle = True)[4]
y1_val = np.load('./samsung/samsung_final_samsung.npy', allow_pickle = True)[5]

y2_train = np.load('./samsung/samsung_final_samsung.npy', allow_pickle = True)[6]
y2_test = np.load('./samsung/samsung_final_samsung.npy', allow_pickle = True)[7]
y2_val = np.load('./samsung/samsung_final_samsung.npy', allow_pickle = True)[8]


from tensorflow.keras.models import load_model
# 제출용!!
model = load_model('./samsung/samsung_final_0.98_91205.hdf5')


result = model.evaluate([x_samsung_test,x_kodex_test],[y1_test,y2_test],batch_size = 8)
print(result)

y_pred = model.predict([x_samsung_test,x_kodex_test])

from sklearn.metrics import r2_score
print('내일 r2 : ', r2_score(y_pred[:,0],y1_test))
print('모레 r2 : ', r2_score(y_pred[:,1],y2_test))

y_pred = model.predict([x_samsung_pred,x_kodex_pred])
print(y_pred)



