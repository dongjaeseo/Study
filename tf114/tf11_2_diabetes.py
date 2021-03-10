from sklearn.datasets import load_diabetes
import tensorflow as tf

dataset = load_diabetes()
# print(dataset.data.shape) (442, 10)
x_data = dataset.data
y_data = dataset.target

x = tf.placeholder(tf.float32, shape = [None, 10])
y = tf.placeholder(tf.float32, shape = [None, 1])

# [실습] 만들거라!!

# 최종 sklearn 의 R2값으로 결론낼것!