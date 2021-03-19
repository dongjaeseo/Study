import numpy as np
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scipy import stats

#0. 변수
batch = 16
seed = 42
dropout = 0.3
epochs = 1000
model_path = '../data/model/lpd_007.hdf5'
save_path = '../data/lpd/sample_007.csv'
save2_path = '../data/lpd/sample_007_2.csv'
sub = pd.read_csv('../data/lpd/sample.csv', header = 0)
es = EarlyStopping(patience = 7)
lr = ReduceLROnPlateau(factor = 0.25, patience = 3, verbose = 1)
cp = ModelCheckpoint(model_path, save_best_only= True)

#1. 데이터
train_gen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range= 0.05,
    height_shift_range= 0.05,
    preprocessing_function= preprocess_input
)

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    zoom_range= 0.1,
    horizontal_flip= True
)

# Found 39000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    '../data/lpd/train_new',
    target_size = (192, 192),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'training'
)

# Found 9000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    '../data/lpd/train_new',
    target_size = (192, 192),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'validation'
)

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    '../data/lpd/test_new',
    target_size = (192, 192),
    class_mode = None,
    batch_size = batch,
    shuffle = False
)
'''
#2. 모델
eff = EfficientNetB4(include_top = False, input_shape=(192, 192, 3))
eff.trainable = True

pretrained = eff.output
globalpooling = GlobalAveragePooling2D()(pretrained)
layer_2 = Dense(1000)(globalpooling)
layer_2 = swish(layer_2)
output = Dense(1000, activation = 'softmax')(layer_2)

model = Model(inputs = eff.input, outputs = output)

#3. 컴파일 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['sparse_categorical_accuracy'])
model.fit(train_data, steps_per_epoch = np.ceil(39000/batch), validation_data= val_data, validation_steps= np.ceil(9000/batch),\
    epochs = epochs, callbacks = [es, cp, lr])
'''
model = load_model(model_path)

#4. 평가 예측
'''
pred = model.predict(test_data, steps = len(test_data))
pred = np.argmax(pred, 1)
print(pred)
sub.loc[:, 'prediction'] = pred
sub.to_csv(save_path, index = False)
'''

# mode 는 제일 많이 반복된 수를 반환 - 모델을 여러번 반복예측 후 높은 반복수를 보인 결과를 저장해보자
# 모델이 짜여졌을때 데이터제너레이터를 사용하여 매번 다른 x 데이터를 통해 예측한다
# tta 포문이 도는동안 매 tta 마다 72000장에 대한 예측을 하고
# result 리스트에 더해준다 >> n tta 를 돈 이후에 result >> (n, 72000)
# temp 변수에 result 를 저장하여
# 한 tta 가 돈 경우 누적된 예측 데이터로 반복횟수가 높은 예측값을 temp_mode 에 저장해준다!
# temp_count 는 mode 의 반복 횟수를 반환하는데 사용한 이유는
# 50 퍼센트 미만의 예측을 반복했다면(프레딕트 값이 변동이 크다면) 어떤 테스트 이미지가 불확실한지 알려준다

result = []
for tta in range(20):
    print(f'{tta} 번째 TTA 진행중')
    pred = model.predict(test_data, steps = len(test_data))
    pred = np.argmax(pred, 1)
    result.append(pred)

    temp = np.array(result)
    temp = np.transpose(result)

    temp_mode = stats.mode(temp, axis = 1).mode
    sub.loc[:, 'prediction'] = temp_mode
    sub.to_csv(f'../data/lpd/submit/sample_007_{tta}.csv', index = False)

    temp_count = stats.mode(temp, axis = 1).count
    for i, count in enumerate(temp_count):
        if count < tta/2.:
            print(f'{i} 번째는 정확도가 {tta/2.} 미만!')

