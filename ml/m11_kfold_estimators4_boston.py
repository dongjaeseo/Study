import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
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
#  [0.56085054 0.74287358 0.64989315 0.7409534  0.75688159]
# AdaBoostRegressor 의 정답률 : 
#  [0.87812514 0.77930873 0.87760829 0.88238083 0.84428766]
# BaggingRegressor 의 정답률 : 
#  [0.85195731 0.82626708 0.89261159 0.77462333 0.81761921]
# BayesianRidge 의 정답률 :
#  [0.63942608 0.71092792 0.82700983 0.7909135  0.51413797]
# CCA 의 정답률 : 
#  [0.69620199 0.59478182 0.63535734 0.69791889 0.70277379]
# DecisionTreeRegressor 의 정답률 : 
#  [0.84721897 0.68623863 0.79991264 0.87302241 0.76116344]
# DummyRegressor 의 정답률 :
#  [-9.08164488e-03 -4.91699313e-03 -1.19916791e-05 -1.60732810e-02
#  -1.32542831e-02]
# ElasticNet 의 정답률 :
#  [0.64686676 0.57747799 0.77986646 0.71360558 0.67254777]
# ElasticNetCV 의 정답률 : 
#  [0.63848488 0.6564454  0.62924945 0.67775657 0.65439822]
# ExtraTreeRegressor 의 정답률 :
#  [0.78121825 0.69232297 0.70518975 0.5458406  0.56751323]
# ExtraTreesRegressor 의 정답률 : 
#  [0.87236225 0.85562585 0.86030847 0.888965   0.92064289]
# GammaRegressor 의 정답률 :
#  [-8.20514900e-05 -1.48058824e-02 -4.32155432e-03 -1.14423979e-03
#  -1.15024094e-02]
# GaussianProcessRegressor 의 정답률 : 
#  [-5.82911156 -5.99554767 -5.65161681 -6.67403634 -5.42177026]
# GeneralizedLinearRegressor 의 정답률 : 
#  [0.73866582 0.68815377 0.62666887 0.58705549 0.71293758]
# GradientBoostingRegressor 의 정답률 : 
#  [0.8623868  0.84707192 0.78553234 0.87372341 0.94757091]
# HistGradientBoostingRegressor 의 정답률 : 
#  [0.91362687 0.87420688 0.82961158 0.83662717 0.79258303]
# HuberRegressor 의 정답률 : 
#  [0.59234782 0.52257697 0.84511898 0.73637402 0.56176577]
# IsotonicRegression 의 정답률 :
#  [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 :
#  [0.40922099 0.30630581 0.68357962 0.46848824 0.29907669]
# KernelRidge 의 정답률 : 
#  [0.77444576 0.76407447 0.60460884 0.56828323 0.74690011]
# Lars 의 정답률 : 
#  [0.74030182 0.72291349 0.71712141 0.69027949 0.63669289]
# LarsCV 의 정답률 : 
#  [0.71956558 0.68384855 0.70047451 0.64993504 0.78327975]
# Lasso 의 정답률 :
#  [0.64823876 0.76713806 0.60519918 0.6755497  0.63451933]
# LassoCV 의 정답률 : 
#  [0.5138081  0.7410016  0.64474247 0.79750473 0.79116249]
# LassoLars 의 정답률 :
#  [-4.44896658e-04 -3.05867084e-05 -2.31881832e-03 -3.72186111e-02
#  -4.10359022e-02]
# LassoLarsCV 의 정답률 : 
#  [0.79507532 0.69708587 0.71237135 0.74693703 0.67370632]
# LassoLarsIC 의 정답률 :
#  [0.69937132 0.76117144 0.58151399 0.70171511 0.78587155]
# LinearRegression 의 정답률 :
#  [0.62711302 0.77061981 0.70181164 0.65856819 0.82330796]
# LinearSVR 의 정답률 :
#  [ 0.53630989  0.56223712 -2.34937137 -0.8496895   0.61919776]
# MLPRegressor 의 정답률 :
#  [0.61252478 0.34514677 0.560391   0.55675045 0.56318213]
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
#  [0.22830149 0.20779535 0.19986635 0.18570163 0.15300406]
# OrthogonalMatchingPursuit 의 정답률 :
#  [0.56561497 0.55991802 0.48988311 0.5580952  0.48324266]
# OrthogonalMatchingPursuitCV 의 정답률 :
#  [0.61083518 0.74847919 0.75472908 0.7094998  0.5950143 ]     
# PLSCanonical 의 정답률 :
#  [-2.67422944 -1.92680596 -0.8909863  -2.18601987 -2.82437708]
# PLSRegression 의 정답률 :
#  [0.74994846 0.67440389 0.64159003 0.6860081  0.72266876]     
# PassiveAggressiveRegressor 의 정답률 :
#  [-0.81743405  0.14006208 -0.40196451  0.28182683 -5.44639375]
# PoissonRegressor 의 정답률 : 
#  [0.81228556 0.77335296 0.79453791 0.74284843 0.70788669]
# RANSACRegressor 의 정답률 : 
#  [ 0.7217071  -1.47195986  0.59076545  0.4075532   0.57739739]
# RadiusNeighborsRegressor 은 없는 놈!!!
# RandomForestRegressor 의 정답률 : 
#  [0.77731805 0.88293033 0.88077778 0.89286716 0.9062601 ]
# RegressorChain 은 없는 놈!!!
# Ridge 의 정답률 :
#  [0.6955077  0.75886838 0.70440319 0.66347639 0.75732558]
# RidgeCV 의 정답률 :
#  [0.71615177 0.70285618 0.65640076 0.80152444 0.61785206]
# SGDRegressor 의 정답률 : 
#  [-1.55218073e+25 -4.40061098e+26 -1.60267229e+25 -1.61456090e+25
#  -1.60908942e+26]
# SVR 의 정답률 : 
#  [0.17820307 0.24774007 0.12647935 0.13427876 0.14967423]
# StackingRegressor 은 없는 놈!!!
# TheilSenRegressor 의 정답률 : 
#  [0.67547448 0.54368708 0.65170773 0.73419379 0.78802671]
# TransformedTargetRegressor 의 정답률 :
#  [0.65874224 0.78912183 0.71416154 0.71585709 0.67364934]
# TweedieRegressor 의 정답률 : 
#  [0.5274766  0.68265845 0.69305834 0.61213885 0.72312379]
# VotingRegressor 은 없는 놈!!!
# _SigmoidCalibration 의 정답률 :
#  [nan nan nan nan nan]