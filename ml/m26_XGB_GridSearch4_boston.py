from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

parameters = [
    {'n_estimators':[100,200,300], 'learning_rate': [0.1, 0.05, 0.01], 'max_depth':[4,5,6]},
    {'n_estimators':[90,200,110], 'learning_rate': [0.1, 0.01, 0.001], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate': [0.1, 0.001, 0.0001], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1], 'colsample_bylebel':[0.6,0.7,0.9]}
]

model = GridSearchCV(XGBRegressor(), parameters, cv = 5)

model.fit(x_train,y_train)

print("모델 스코어는 : ", model.score(x_test,y_test))
print("최적의 모델은? : ", model.best_estimator_)