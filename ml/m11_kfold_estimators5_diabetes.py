import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, random_state = 33)

kfold = KFold(n_splits=5, shuffle = True)
allAlgorithms = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        score = cross_val_score(model,x_train,y_train, cv = kfold)
        print(name, '의 정답률 : \n', score)
    except: 
        # continue
        print(name, '은 없는 놈!!!')

# ARDRegression 의 정답률 : 
#  [0.45027407 0.5805254  0.51464317 0.43171075 0.28927513]
# AdaBoostRegressor 의 정답률 : 
#  [0.49219182 0.47599021 0.3001358  0.27689004 0.44614619]
# BaggingRegressor 의 정답률 : 
#  [0.32359352 0.19846417 0.47147452 0.41426372 0.48152978]
# BayesianRidge 의 정답률 :
#  [0.5734255  0.40963756 0.43377087 0.4113517  0.52485761]
# CCA 의 정답률 : 
#  [0.6043498  0.45252738 0.57631094 0.40677534 0.31919013]
# DecisionTreeRegressor 의 정답률 :
#  [-0.16027437  0.04746291 -0.08517332  0.12274555 -0.3772114 ]
# DummyRegressor 의 정답률 : 
#  [-0.0055111  -0.0038017  -0.03147145 -0.00143847 -0.07965935]
# ElasticNet 의 정답률 :
#  [ 0.00667842 -0.00101986 -0.00862449  0.00609041  0.00535498]
# ElasticNetCV 의 정답률 : 
#  [0.34572225 0.52811837 0.44902462 0.317557   0.43422064]
# ExtraTreeRegressor 의 정답률 :
#  [ 0.11810794 -0.04511682  0.1711481  -0.05968451 -0.50182065]
# ExtraTreesRegressor 의 정답률 : 
#  [0.47706462 0.42979412 0.21665019 0.42277169 0.43870328]
# GammaRegressor 의 정답률 :
#  [-0.00012314 -0.01093497 -0.01712263  0.00456229 -0.0090571 ]
# GaussianProcessRegressor 의 정답률 : 
#  [-19.37587305  -7.88040254 -19.92809952 -16.10982801 -21.70597945]
# GeneralizedLinearRegressor 의 정답률 :
#  [-0.00646536  0.00565009 -0.01151466 -0.00047323 -0.00655725]
# GradientBoostingRegressor 의 정답률 :
#  [0.55674885 0.33436826 0.33702134 0.35959096 0.40396229]
# HistGradientBoostingRegressor 의 정답률 :
#  [0.4251987  0.31789002 0.32801263 0.42291024 0.39087813]
# HuberRegressor 의 정답률 :
#  [0.43616168 0.41429654 0.58064946 0.49627685 0.4624234 ]
# IsotonicRegression 의 정답률 :
#  [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 :
#  [0.3702605  0.45280924 0.31619013 0.13454506 0.48410554]
# KernelRidge 의 정답률 :
#  [-3.65688665 -3.12465299 -4.22690097 -3.94466845 -4.12079897]
# Lars 의 정답률 :
#  [ 0.32339153  0.59871696  0.35916615 -1.61849312  0.49892356]
# LarsCV 의 정답률 :
#  [0.455951   0.32076927 0.57576048 0.46964144 0.42699369]
# Lasso 의 정답률 :
#  [0.28235683 0.23421587 0.34306616 0.38255113 0.34219739]
# LassoCV 의 정답률 :
#  [0.59756856 0.35608127 0.50360032 0.4390567  0.45313416]
# LassoLars 의 정답률 :
#  [0.11341431 0.36101415 0.35953599 0.42925473 0.31760748]
# LassoLarsCV 의 정답률 :
#  [0.50648509 0.61041482 0.47659302 0.41004136 0.41651501]
# LassoLarsIC 의 정답률 :
#  [0.49774991 0.32643615 0.58275875 0.3166914  0.43003506]     
# LinearRegression 의 정답률 :
#  [0.41877836 0.48923319 0.28984584 0.46718309 0.58444579]     
# LinearSVR 의 정답률 :
#  [-0.37330976 -0.55811137 -0.78361203 -0.23325095 -0.56664929]
# MLPRegressor 의 정답률 :
#  [-4.7079154  -2.81543853 -2.41350546 -3.24374289 -2.85809442]
# MultiOutputRegressor 은 없는 놈!!!
# MultiTaskElasticNet 의 정답률 :
#  [nan nan nan nan nan]
# MultiTaskElasticNetCV 의 정답률 : 
#  [nan nan nan nan nan]
# MultiTaskLasso 의 정답률 :
#  [nan nan nan nan nan]
# MultiTaskLassoCV 의 정답률 :
#  [nan nan nan nan nan]
# NuSVR 의 정답률 : 
#  [0.10629355 0.1390872  0.11443869 0.1334054  0.13071553]
# OrthogonalMatchingPursuit 의 정답률 :
#  [0.34785673 0.20494118 0.1975019  0.28491052 0.31967326]
# OrthogonalMatchingPursuitCV 의 정답률 : 
#  [0.60730541 0.42798094 0.4003985  0.34513354 0.46488435]
# PLSCanonical 의 정답률 : 
# PLSRegression 의 정답률 :
#  [0.52863587 0.3992796  0.31294243 0.60193958 0.37189346]
# PassiveAggressiveRegressor 의 정답률 :
#  [0.34405711 0.48097707 0.37667777 0.34254512 0.52729554]
# PoissonRegressor 의 정답률 :
#  [0.3082668  0.35292995 0.35205176 0.26742324 0.29648042]
# RANSACRegressor 의 정답률 :
#  [ 0.02930885  0.31119408  0.46273596 -0.1946125   0.17222404]
# RadiusNeighborsRegressor 의 정답률 :
#  [-0.0002057  -0.15941855 -0.02344504 -0.00366405 -0.06334314]
# RandomForestRegressor 의 정답률 :
#  [0.43762112 0.45791667 0.56524654 0.30981825 0.22497601]
# RegressorChain 은 없는 놈!!!
# Ridge 의 정답률 :
#  [0.3028598  0.3918411  0.43081426 0.29118474 0.45244898]
# RidgeCV 의 정답률 :
#  [0.57600723 0.32944455 0.47804924 0.44757564 0.45812645]
# SGDRegressor 의 정답률 :
#  [0.26911728 0.32055032 0.48729403 0.37238674 0.38361196]
# SVR 의 정답률 :
#  [0.14836037 0.15701991 0.02625176 0.12279234 0.13201579]
# StackingRegressor 은 없는 놈!!!
# TheilSenRegressor 의 정답률 :
#  [0.52448078 0.52498101 0.29547596 0.35438816 0.42554706]
# TransformedTargetRegressor 의 정답률 :
#  [0.42495912 0.41765626 0.52133028 0.47693097 0.43845839]
# TweedieRegressor 의 정답률 :
#  [ 0.00057085 -0.00957915  0.00415182  0.00578395  0.00541371]
# VotingRegressor 은 없는 놈!!!
# _SigmoidCalibration 의 정답률 :
#  [nan nan nan nan nan]