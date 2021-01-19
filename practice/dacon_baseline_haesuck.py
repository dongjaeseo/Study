import pandas as pd
import numpy as np
import tensorflow as tf

# train 파일, submission 파일 불러오기
train = pd.read_csv('./practice/dacon/data/train/train.csv')
sub = pd.read_csv('./practice/dacon/data/sample_submission.csv')

# preprocess_data() 라는 함수를 정의함
# data 에서 마지막 하루만 리턴
def preprocess_data(data):
        temp = data.copy()
        return temp.iloc[-48:, :]

df_test = []

# 81개의 테스트 파일에서 마지막 하루만 전부 붙인다
for i in range(81):
    file_path = './practice/dacon/data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp)
    # temp.shape = (1, 48, 9)
    df_test.append(temp)

# 전부 한 데이터프레임으로 이어붙여줌
X_test = pd.concat(df_test)
# 마지막 이틀치 데이터를 뒤에 또 붙여줌
## 결과적으로 X_test 에는 81개의 test 파일(7,48,9)의 각 마지막 일데이터(1,48,9) 를 전부 한 데이터프레임으로 이어주고
## 그 끝에 마지막 이틀치를 복붙해준다 >> (81+2)*48,9 만큼의 쉐이프를 가진다
X_test = X_test.append(X_test[-96:])

# Hour - 시간
# Minute - 분
# DHI - 수평면 산란일사량(Diffuse Horizontal Irradiance (W/m2))
# DNI - 직달일사량(Direct Normal Irradiance (W/m2))
# WS - 풍속(Wind Speed (m/s))
# RH - 상대습도(Relative Humidity (%))
# T - 기온(Temperature (Degree C))
# Target - 태양광 발전량 (kW)

## Add_features 라는 함수를 만들건데
# 여러 상수를 이용하여 DHI, DNI 보다 더 직관적인 GHI 라는 피처를 만들거다

def Add_features(data):
  c = 243.12
  b = 17.62
  gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
  dp = ( c * gamma) / (b - gamma)
  data.insert(1,'Td',dp)
  data.insert(1,'T-Td',data['T']-data['Td'])
  data.insert(1,'GHI',data['DNI']+data['DHI'])
  return data
## 데이터의 새로운 칼럼 순서 => GHI T-Td Td   day hour minute DHI DNI WS RH T Target   // 12개!!


# train, X_test 에 새로운 칼럼 추가해주고
train = Add_features(train)
X_test = Add_features(X_test)

# 일, 분은 큰 연관이 없으므로 드랍 // 데이터 칼럼은 이제 10개!!
df_train = train.drop(['Day','Minute'],axis=1)
df_test  = X_test.drop(['Day','Minute'],axis=1)

# column_indices 에 10개의 칼럼명을 딕셔너리 형태로 저장
column_indices = {name: i for i, name in enumerate(df_train.columns)}

#Train and Validation split
n = len(train) # (1095*48) = 52560
# train, val 로 나눠주는데 => 근데 이렇게 하면 일 단위로 깔끔하게 안잘릴텐데,,,,??
train_df = df_train[0:int(n*0.8)] 
val_df   = df_train[int(n*0.8):]
test_df = df_test

# Normalization
num_features = train_df.shape[1]    # 10!!

train_mean = train_df.mean() # (10,) >> 각 열의 평균값이다
train_std  = train_df.std() # 각 열의 표준편차값!!

train_df = (train_df - train_mean) / train_std
val_df   =  (val_df - train_mean) / train_std
test_df  = (test_df - train_mean) / train_std  ## 전처리 해주는건가봄

## 뭔소린지
'''
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        #Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        #Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
'''
# 위에랑 연결되는듯
'''
def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
    # Slicing doesn't preserve static shape information, so set the shapes
    # Manually. This way the tf.data.Datasets' are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    return inputs, labels

WindowGenerator.split_window = split_window
'''

# def make_dataset(self, data,is_train=True):
#     data = np.array(data, dtype=np.float32)
#     if is_train==True:
#         ds = tf.keras.preprocessing.timeseries_dataset_from_array(
#                     data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, shuffle=True, batch_size=256,)
#     else:
#         ds = tf.keras.preprocessing.timeseries_dataset_from_array(
#                     data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, shuffle=False, batch_size=256,)
#     ds = ds.map(self.split_window)
#     return ds

# WindowGenerator.make_dataset = make_dataset

# WindowGenerator 들어가는거 걍 다 안씀~~ >> 시각화 하기 위한것같음

import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

# quantile_loss 라는 함수를 만들건데 예측값, 실제값을 넣으면 
# 실제값이 더 클경우 q를, 예측값이 더 클 경우 1-q 를 곱합만큼의 절대값을 주고 그 전체 합의 평균을 준다
# axis -1 은 텐서의 제일 안쪽 데이터를 다룰것이란 얘기
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

OUT_STEPS = 96

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')

# quantile_plot 안가져옴 이것도 시각화의 일부

def DenseModel():
    model = tf.keras.Sequential()
    model.add(L.Lambda(lambda x: x[:, -1:, :]))
    model.add(L.Dense(512, activation='relu'))
    model.add(L.Dense(256, activation='relu'))
    model.add(L.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros))
    model.add(L.Reshape([OUT_STEPS, num_features]))
    return model

# 빈 데이터 프레임을 만들어두고
Dense_actual_pred = pd.DataFrame()
Dense_val_score = pd.DataFrame()
'''
# for 문 이렇게도 돌리는구낭
for q in quantiles:
	model = DenseModel()
	model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(q, y, pred)])
	history = model.fit(w1.train, validation_data=w1.val, epochs=20, callbacks=[early_stopping])
	pred = model.predict(w1.test, verbose=0)
	target_pred = pd.Series(pred[::48][:,:,9].reshape(7776)) #Save predicted value (striding=48 step, 9 = TARGET) 
	Dense_actual_pred = pd.concat([Dense_actual_pred,target_pred],axis=1)
	Dense_val_score[f'{q}'] = model.evaluate(w1.val)
	w1.quantile_plot(model, quantile=q)

Dense_actual_pred.columns = quantiles
#Denormalizing TARGET values
Dense_actual_pred_denorm = Dense_actual_pred*train_std['TARGET'] + train_mean['TARGET']
#Replace Negative value to Zero
Dense_actual_pred_nn = np.where(Dense_actual_pred_denorm<0, 0, Dense_actual_pred_denorm)

sub.iloc[:,1:] = Dense_actual_pred_nn
sub.to_csv("/content/drive/MyDrive/data_solar/submission/submission_210114_quantile_dense_nn.csv",index=False)

### AutoRegressive LSTM
class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_features)

feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

def warmup(self, inputs):
      # inputs.shape => (batch, time, features)
  # x.shape => (batch, lstm_units)
  x, *state = self.lstm_rnn(inputs)
  # predictions.shape => (batch, features)
  prediction = self.dense(x)
  return prediction, state

FeedBack.warmup = warmup

prediction, state = feedback_model.warmup(w1.example[0])
prediction.shape
'''