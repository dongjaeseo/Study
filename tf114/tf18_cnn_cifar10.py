import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
tf.compat.v1.set_random_seed(66)

tf.compat.v1.disable_eager_execution()

print(tf.executing_eagerly()) # False # False 위 코드를 실행하면 펄스가 된다
# 즉 텐서플로 2 버전에서도 1처럼 코딩할 수 있다


print(tf.__version__)


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255.
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255.

training_epochs = 15
batch_size = 64
total_batch = int(len(x_train)/batch_size) # 60000 / 100

x = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#2. 모델구성

# L1.
w1 = tf.compat.v1.get_variable('w1', shape = [3, 3, 3, 128])
L1 = tf.nn.conv2d(x, w1, strides = [1,1,1,1], padding = 'SAME')
# Conv2D(filter, kernel_size, input_shape) 서머리???
# Conv2D(10, (2, 2), input_shape = (7, 7, 1)) 파라미터의 개수?
L1 = tf.compat.v1.layers.batch_normalization(L1)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print(L1)

# L2.
w2 = tf.compat.v1.get_variable('w2', shape = [3, 3, 128, 64])
L2 = tf.nn.conv2d(L1, w2, strides = [1,1,1,1], padding = 'SAME')
L2 = tf.compat.v1.layers.batch_normalization(L2)
L2 = tf.nn.relu(L2)
# L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print(L2)

# L3.
w3 = tf.compat.v1.get_variable('w3', shape = [3, 3, 64, 64])
L3 = tf.nn.conv2d(L2, w3, strides = [1,1,1,1], padding = 'SAME')
L3 = tf.compat.v1.layers.batch_normalization(L3)
L3 = tf.nn.relu(L3)
# L3 = tf.nn.max_pool(L3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print(L3)

# L4.
w4 = tf.compat.v1.get_variable('w4', shape = [3, 3, 64, 32])
L4 = tf.nn.conv2d(L3, w4, strides = [1,1,1,1], padding = 'SAME')
L4 = tf.compat.v1.layers.batch_normalization(L4)
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print(L4) # Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

# Flatten
L_flat = tf.reshape(L4, [-1, 8*8*32])
print(L_flat) # Tensor("Reshape:0", shape=(?, 256), dtype=float32)

# L5.
w5 = tf.compat.v1.get_variable('w5', shape = [8*8*32, 64], initializer = tf.keras.initializers.HeNormal())
b5 = tf.Variable(tf.ones([64]), name = 'b5')/100.
L5 = tf.nn.relu(tf.matmul(L_flat, w5) + b5)
L5 = tf.nn.dropout(L5, rate = 0.2) 
print(L5) # Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

# L6.
w6 = tf.compat.v1.get_variable('w6', shape = [64, 32], initializer = tf.keras.initializers.HeNormal())
b6 = tf.Variable(tf.ones([32]), name = 'b6')/100.
L6 = tf.nn.relu(tf.matmul(L5, w6) + b6)
# L6 = tf.nn.dropout(L6, rate = 0.2) 
print(L6) # Tensor("dropout_1/mul_1:0", shape=(?, 32), dtype=float32)

# L7.
w7 = tf.compat.v1.get_variable('w7', shape = [32, 10], initializer = tf.keras.initializers.HeNormal())
b7 = tf.Variable(tf.ones([10]), name = 'b7')/100.
hypothesis = tf.nn.softmax(tf.matmul(L6, w7) + b7)
print(hypothesis) # Tensor("Softmax:0", shape=(?, 10), dtype=float32)

#3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.log(hypothesis), 1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 4*1e-4)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss = 0

    for i in range(total_batch):  # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, train], feed_dict = feed_dict)
        avg_loss += c/total_batch
    
    prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict = {x:x_test, y:y_test})
    print(f'Epoch {epoch} \t===========>\t loss : {avg_loss:.8f}\t acc : {acc}')

print('훈련 끝')

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc : ', sess.run(accuracy, feed_dict = {x:x_test, y:y_test}))

