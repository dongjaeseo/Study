import numpy as np
from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer, load_wine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
# mnist, fashion, cifar10, cifar100

# boston
boston_dataset = load_boston()
boston_x = boston_dataset['data']
boston_y = boston_dataset['target']

np.save('./data/boston_x.npy', arr = boston_x)
np.save('./data/boston_y.npy', arr = boston_y)


# diabetes
diabetes_dataset = load_diabetes()
diabetes_x = diabetes_dataset['data']
diabetes_y = diabetes_dataset['target']

np.save('./data/diabetes_x.npy', arr = diabetes_x)
np.save('./data/diabetes_y.npy', arr = diabetes_y)


# breast_cancer
breast_cancer_dataset = load_breast_cancer()
breast_cancer_x = breast_cancer_dataset['data']
breast_cancer_y = breast_cancer_dataset['target']

np.save('./data/cancer_x.npy', arr = breast_cancer_x)
np.save('./data/cancer_y.npy', arr = breast_cancer_y)


# wine
wine_dataset = load_wine()
wine_x = wine_dataset['data']
wine_y = wine_dataset['target']

np.save('./data/wine_x.npy', arr = wine_x)
np.save('./data/wine_y.npy', arr = wine_y)


# mnist
(m_x_train,m_y_train),(m_x_test,m_y_test) = mnist.load_data()
np.save('./data/mnist_x_train.npy', arr = m_x_train)
np.save('./data/mnist_x_test.npy', arr = m_x_test)
np.save('./data/mnist_y_train.npy', arr = m_y_train)
np.save('./data/mnist_y_test.npy', arr = m_y_test)

# fashion_mnist
(f_x_train,f_y_train),(f_x_test,f_y_test) = fashion_mnist.load_data()
np.save('./data/fashion_mnist_x_train.npy', arr = f_x_train)
np.save('./data/fashion_mnist_x_test.npy', arr = f_x_test)
np.save('./data/fashion_mnist_y_train.npy', arr = f_y_train)
np.save('./data/fashion_mnist_y_test.npy', arr = f_y_test)

# cifar10
(c10_x_train,c10_y_train),(c10_x_test,c10_y_test) = cifar10.load_data()
np.save('./data/cifar10_x_train.npy', arr = c10_x_train)
np.save('./data/cifar10_x_test.npy', arr = c10_x_test)
np.save('./data/cifar10_y_train.npy', arr = c10_y_train)
np.save('./data/cifar10_y_test.npy', arr = c10_y_test)

# cifar100
(c10_x_train,c10_y_train),(c10_x_test,c10_y_test) = cifar100.load_data()
np.save('./data/cifar100_x_train.npy', arr = c10_x_train)
np.save('./data/cifar100_x_test.npy', arr = c10_x_test)
np.save('./data/cifar100_y_train.npy', arr = c10_y_train)
np.save('./data/cifar100_y_test.npy', arr = c10_y_test)