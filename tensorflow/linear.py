import tensorflow as tf

x_data = [1.,2.,3.]
y_data = [1.,2.,3.]

w= tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b= tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)




hypothesis = w*X + b

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step%20==0:
        print(step, sess.run(cost, feed_dict={X : x_data, Y : y_data}), sess.run(w), sess.run(b))


for preX in range(5, 100):
    print("예축 결과----", sess.run(hypothesis, feed_dict={X: preX}))


