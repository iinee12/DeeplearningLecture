import tensorflow as tf
import numpy as np

xy = np.loadtxt('../data/03train.txt', dtype='float32')

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


W= tf.Variable(tf.random_uniform([3,1],-1., 1.))

hypothesis = tf.matmul(x_data, W)

cost = tf.reduce_mean(tf.square(hypothesis-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0 :
        print(step, sess.run(cost), sess.run(W))