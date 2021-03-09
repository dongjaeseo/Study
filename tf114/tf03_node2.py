# [실습]
# 덧셈
# 뺄셈
# 곱셈
# 나눗셈
# 맹그러

import tensorflow as tf
import warnings

node1 = tf.constant(12.0)
node2 = tf.constant(5.0)

sess = tf.compat.v1.Session()

# 덧셈
node_add = tf.add(node1, node2)
print('덧셈 :', sess.run(node_add))

# 뺄셈
node_subtract = tf.subtract(node1, node2)
print('뺄셈 :', sess.run(node_subtract))

# 곱셈
node_multiply = tf.multiply(node1, node2)
print('곱셈 :', sess.run(node_multiply))

# 나눗셈
node_divide = tf.divide(node1, node2)
print('나눗셈 :', sess.run(node_divide))

# 나머지
node_mod = tf.math.mod(node1, node2)
print('나머지 :', sess.run(node_mod))