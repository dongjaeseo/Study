import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, random_state = 33)

kfold = KFold(n_splits=5, shuffle = True)
allAlgorithms = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        score = cross_val_score(model,x_train,y_train, cv = kfold)
        print(name, '의 정답률 : \n', score)
    except: 
        # continue
        print(name, '은 없는 놈!!!')

# AdaBoostClassifier 의 정답률 :
#  [0.96703297 0.96703297 0.94505495 0.96703297 0.94505495]
# BaggingClassifier 의 정답률 :
#  [0.96703297 0.91208791 0.94505495 0.98901099 0.94505495]
# BernoulliNB 의 정답률 :
#  [0.68131868 0.6043956  0.65934066 0.6043956  0.59340659]
# CalibratedClassifierCV 의 정답률 :
#  [0.93406593 0.92307692 0.92307692 0.89010989 0.95604396]
# CategoricalNB 은 없는 놈!!!
# CheckingClassifier 의 정답률 :
#  [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 놈!!!
# ComplementNB 의 정답률 :
#  [0.95604396 0.9010989  0.92307692 0.87912088 0.89010989]
# DecisionTreeClassifier 의 정답률 :
#  [0.92307692 0.92307692 0.98901099 0.93406593 0.95604396]
# DummyClassifier 의 정답률 :
#  [0.49450549 0.53846154 0.59340659 0.47252747 0.52747253]
# ExtraTreeClassifier 의 정답률 :
#  [0.93406593 0.95604396 0.92307692 0.93406593 0.93406593]
# ExtraTreesClassifier 의 정답률 :
#  [1.         0.96703297 0.97802198 0.93406593 0.94505495]
# GaussianNB 의 정답률 :
#  [0.91208791 0.96703297 0.91208791 0.96703297 0.94505495]
# GaussianProcessClassifier 의 정답률 : 
#  [0.91208791 0.92307692 0.98901099 0.89010989 0.92307692]
# GradientBoostingClassifier 의 정답률 : 
#  [0.98901099 0.94505495 0.96703297 0.97802198 0.94505495]
# HistGradientBoostingClassifier 의 정답률 : 
#  [0.95604396 0.95604396 0.97802198 0.95604396 0.93406593]
# KNeighborsClassifier 의 정답률 :
#  [0.86813187 0.94505495 0.98901099 0.91208791 0.92307692]
# LabelPropagation 의 정답률 : 
#  [0.24175824 0.41758242 0.48351648 0.3956044  0.40659341]
# LabelSpreading 의 정답률 : 
#  [0.32967033 0.37362637 0.38461538 0.40659341 0.42857143]
# LinearDiscriminantAnalysis 의 정답률 :
#  [0.96703297 0.96703297 0.92307692 0.96703297 0.94505495]
# LinearSVC 의 정답률 : 
#  [0.95604396 0.8021978  0.9010989  0.65934066 0.93406593]
# LogisticRegression 의 정답률 : 
#  [0.95604396 0.94505495 0.91208791 0.96703297 0.91208791]
# LogisticRegressionCV 의 정답률 : 
#  [0.93406593 0.98901099 0.94505495 0.93406593 0.96703297]
# MLPClassifier 의 정답률 : 
#  [0.91208791 0.95604396 0.93406593 0.91208791 0.93406593]
# MultiOutputClassifier 은 없는 놈!!!
# MultinomialNB 의 정답률 :
#  [0.92307692 0.89010989 0.87912088 0.9010989  0.92307692]
# NearestCentroid 의 정답률 : 
#  [0.93406593 0.89010989 0.85714286 0.89010989 0.9010989 ]
# NuSVC 의 정답률 : 
#  [0.87912088 0.9010989  0.89010989 0.86813187 0.87912088]
# OneVsOneClassifier 은 없는 놈!!!
# OneVsRestClassifier 은 없는 놈!!!
# OutputCodeClassifier 은 없는 놈!!!
# PassiveAggressiveClassifier 의 정답률 :
#  [0.48351648 0.93406593 0.9010989  0.92307692 0.76923077]
# Perceptron 의 정답률 :
#  [0.86813187 0.91208791 0.85714286 0.9010989  0.96703297]
# QuadraticDiscriminantAnalysis 의 정답률 :
#  [0.98901099 0.95604396 0.94505495 0.93406593 0.96703297]
# RadiusNeighborsClassifier 은 없는 놈!!!
# RandomForestClassifier 의 정답률 : 
#  [0.94505495 0.96703297 0.96703297 0.94505495 0.96703297]
# RidgeClassifier 의 정답률 :
#  [0.96703297 0.97802198 0.93406593 0.94505495 0.92307692]
# RidgeClassifierCV 의 정답률 : 
#  [0.95604396 0.96703297 0.95604396 0.94505495 0.94505495]
# SGDClassifier 의 정답률 : 
#  [0.95604396 0.86813187 0.89010989 0.8021978  0.9010989 ]
# SVC 의 정답률 : 
#  [0.91208791 0.87912088 0.94505495 0.91208791 0.94505495]
# StackingClassifier 은 없는 놈!!!
# VotingClassifier 은 없는 놈!!!