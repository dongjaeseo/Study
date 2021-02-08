# 61카피해서
#1. model.cv_results 를 붙여서 완성

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

#2. 모델
def build_model(drop=0.5,optimizer = 'adam'):
    inputs = Input(shape = (28*28,), name = 'input')
    x = Dense(512, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer, metrics = ['acc'], loss = 'categorical_crossentropy')

    return model

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {'batch_size' : batches, 'optimizer' : optimizers, 'drop' : dropout}
hyperparameters = create_hyperparameter()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn= build_model, verbose = 1)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# search = RandomizedSearchCV(model2, hyperparameters, cv = 3)
search = GridSearchCV(model2, hyperparameters, cv = 3)

search.fit(x_train,y_train,verbose = 1)

print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
acc = search.score(x_test, y_test)
print('최종 스코어 : ', acc)
print(search.cv_results_)

## search.cv_results_ 는 각각의 파라미터에 대한 메트릭스 값을 리턴한다!
## 이 중 메인컬럼은 mean_test_score 인데 그리드 서치에서의 cv값이 3이므로 스플릿0, 스플릿1, 스플릿2 의 테스트 점수의 평균이다!
## 이 값을 이용하여 각각의 파라미터의 성능을 비교해볼 수 있다!
## std_test_score 는 스플릿 0, 1, 2의 standard deviation 을 출력하는데
## 이 값을 이용하여 파라미터가 얼마나 꾸준히? 영향을 미치는지(0,1,2의 deviation이 아무래도 작아야 좋겠징??) 알 수있다!!
## 그리고 time 칼럼들은 각각의 파라미터마다 걸린 시간을 알 수있다!