# [실습] 만들거라!!
# 최종 sklearn 의 R2값으로 결론낼것!

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(42)

dataset = load_breast_cancer()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, random_state= 42)

x = tf.placeholder(tf.float32, shape = [None, 30])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random.normal([30,1], mean = 0, stddev= 0.01), name = 'weight')
b = tf.Variable(tf.random.normal([1], mean = 0, stddev= 0.01), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

train = tf.train.AdamOptimizer(learning_rate= 2*1e-7).minimize(loss)
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(w))
    for epoch in range(73001):
        _, train_loss, train_w = sess.run([train, loss, w], feed_dict= {x:x_train, y:y_train})
        if epoch%100 == 0:
            print(f'Epoch {epoch} === Loss {train_loss}')
    
    a = sess.run(accuracy, feed_dict= {x:x_test, y:y_test})
    print('Accuracy : ', a)
    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = np.where(y_pred>0.5, 1, 0)
    print('Acc_score : ', accuracy_score(y_test, y_pred))

# Accuracy :  0.9385965
# Acc_score :  0.9385964912280702