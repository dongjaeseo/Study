import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
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
#  [0.31034483 0.96551724 0.96428571 0.75       0.92857143]
# BaggingClassifier 의 정답률 : 
#  [0.93103448 0.89655172 1.         0.96428571 0.78571429]
# BernoulliNB 의 정답률 :
#  [0.37931034 0.37931034 0.53571429 0.21428571 0.14285714]
# CalibratedClassifierCV 의 정답률 : 
#  [1.         0.82758621 0.82142857 0.85714286 0.82142857]
# CategoricalNB 은 없는 놈!!!
# CheckingClassifier 의 정답률 :
#  [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 놈!!!
# ComplementNB 의 정답률 :
#  [0.55172414 0.75862069 0.64285714 0.64285714 0.75      ]
# DecisionTreeClassifier 의 정답률 :
#  [0.86206897 0.82758621 0.89285714 0.96428571 0.96428571]
# DummyClassifier 의 정답률 : 
#  [0.37931034 0.4137931  0.42857143 0.32142857 0.35714286]
# ExtraTreeClassifier 의 정답률 :
#  [0.75862069 0.96551724 0.82142857 0.78571429 0.89285714]
# ExtraTreesClassifier 의 정답률 : 
#  [1.         1.         0.92857143 1.         1.        ]
# GaussianNB 의 정답률 :
#  [1.         1.         0.96428571 1.         0.92857143]
# GaussianProcessClassifier 의 정답률 : 
#  [0.55172414 0.31034483 0.32142857 0.42857143 0.42857143]
# GradientBoostingClassifier 의 정답률 : 
#  [0.89655172 0.89655172 0.92857143 0.96428571 0.96428571]
# HistGradientBoostingClassifier 의 정답률 : 
#  [1.         0.96551724 1.         0.89285714 0.96428571]
# KNeighborsClassifier 의 정답률 :
#  [0.79310345 0.75862069 0.82142857 0.75       0.67857143]
# LabelPropagation 의 정답률 :
#  [0.48275862 0.44827586 0.32142857 0.60714286 0.57142857]
# LabelSpreading 의 정답률 : 
#  [0.4137931  0.34482759 0.5        0.57142857 0.46428571]
# LinearDiscriminantAnalysis 의 정답률 : 
#  [0.96551724 1.         0.96428571 0.92857143 1.        ]
# LinearSVC 의 정답률 : 
#  [0.89655172 0.93103448 0.75       0.96428571 0.96428571]
# LogisticRegression 의 정답률 : 
#  [0.89655172 0.96551724 0.96428571 0.92857143 0.96428571]
# LogisticRegressionCV 의 정답률 : 
#  [1.         0.96551724 0.92857143 0.92857143 0.96428571]
# MLPClassifier 의 정답률 : 
#  [0.5862069  0.96551724 0.89285714 0.42857143 0.07142857]
# MultiOutputClassifier 은 없는 놈!!!
# MultinomialNB 의 정답률 :
#  [0.89655172 0.82758621 0.96428571 0.85714286 0.92857143]
# NearestCentroid 의 정답률 : 
#  [0.68965517 0.79310345 0.64285714 0.85714286 0.60714286]
# NuSVC 의 정답률 :
#  [0.96551724 0.75862069 0.82142857 0.92857143 0.85714286]
# OneVsOneClassifier 은 없는 놈!!!
# OneVsRestClassifier 은 없는 놈!!!
# OutputCodeClassifier 은 없는 놈!!!
# PassiveAggressiveClassifier 의 정답률 :
#  [0.62068966 0.37931034 0.53571429 0.71428571 0.14285714]
# Perceptron 의 정답률 :
#  [0.55172414 0.65517241 0.60714286 0.35714286 0.53571429]
# QuadraticDiscriminantAnalysis 의 정답률 :
#  [1.         1.         0.92857143 1.         0.96428571]
# RadiusNeighborsClassifier 은 없는 놈!!!
# RandomForestClassifier 의 정답률 : 
#  [0.96551724 1.         0.92857143 0.96428571 1.        ]
# RidgeClassifier 의 정답률 :
#  [1.         0.93103448 1.         0.96428571 1.        ]
# RidgeClassifierCV 의 정답률 :
#  [1. 1. 1. 1. 1.]
# SGDClassifier 의 정답률 :
#  [0.44827586 0.75862069 0.53571429 0.60714286 0.64285714]
# SVC 의 정답률 :
#  [0.72413793 0.48275862 0.64285714 0.82142857 0.57142857]
# StackingClassifier 은 없는 놈!!!
# VotingClassifier 은 없는 놈!!!