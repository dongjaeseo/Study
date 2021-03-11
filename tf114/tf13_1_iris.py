import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
tf.set_random_seed(66)

dataset = load_iris()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, random_state= 66)

encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()

x = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.float32, [None, 3])

w = tf.Variable(tf.random.normal([4,3]), name = 'weight')
b = tf.Variable(tf.random.normal([1,3]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

train = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(10001):
        cur_loss, _ = sess.run([loss, train], feed_dict = {x:x_train, y:y_train})
        if epoch%20 == 0:
            print(f'Epoch {epoch} ==== {cur_loss}')
    
    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    print('\naccuracy_score : ', accuracy_score(y_test, y_pred))

# accuracy_score :  1.0