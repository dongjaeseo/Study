import numpy as np
import pandas as pd
import os
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, GlobalAveragePooling2D, Input, GaussianDropout
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scipy import stats

#0. 변수
filenum = 29
img_size = 192
batch = 16
seed = 42
epochs = 1000
test_dir = '../data/lpd/test_new3'
model_path = '../data/model/lpd_025.hdf5'
save_folder = '../data/lpd/submit_{0:03}'.format(filenum)
# sub = pd.read_csv('../data/lpd/sample.csv', header = 0)
sub = pd.read_csv('../data/lpd/submit_022/sample_022_50.csv', header = 0)

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

x_pred_idx = np.load('../data/lpd/below_threshold.npy')
print(x_pred_idx.shape)
#1. 데이터

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    width_shift_range= 0.05,
    height_shift_range= 0.05
)

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size = (img_size, img_size),
    class_mode = None,
    batch_size = batch,
    shuffle = False
)

#2. 모델
model = load_model(model_path)

#4. 평가 예측

cumsum = np.zeros([x_pred_idx.shape[0], 1000])
count_result = []
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - TTA')
    pred = model.predict(test_data, steps = len(test_data), verbose = True) # (72000, 1000)
    pred = np.array(pred)
    cumsum = np.add(cumsum, pred)
    temp = cumsum / (tta+1)
    temp_sub = np.argmax(temp, 1)
    temp_percent = np.max(temp, 1)
    
    count = 0
    i = 0
    for percent in temp_percent:
        if percent < 0.3:
            print(f'{i} 번째 테스트 이미지는 {percent}% 의 정확도를 가짐')
            count += 1
        i += 1
    print(f'TTA {tta+1} : {count} 개가 불확실!')
    
    count_result.append(count)
    print(f'기록 : {count_result}')
    
    for i, idx in enumerate(x_pred_idx):
        sub.iloc[idx, 1] = temp_sub[i]

    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(filenum, (tta+1)), index = False)
