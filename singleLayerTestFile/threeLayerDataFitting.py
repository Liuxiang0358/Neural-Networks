import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, outsize, activation_func = None):
    w = tf.Variable(tf.random_normal([in_size, outsize]))
    b = tf.Variable(tf.zeros([1, outsize]) + 0.1)
    y = tf.matmul(inputs, w) + b
    if activation_func is not None:
        y = activation_func(y)
    return y

xs = tf.placeholder('float32', [None, 1])
ys = tf.placeholder('float32', [None, 1])


x_data = np.linspace(-1, 1, 8000)[:, np.newaxis]
y_realData = np.square(x_data) - 0.5
y_data = y_realData + np.random.normal(loc=0, scale=0.18, size=x_data.shape)


l1 = add_layer(xs, 1, 10, tf.nn.relu)
l2 = add_layer(l1, 10, 50, tf.nn.relu)

output = add_layer(l2, 50, 1)
loss = tf.reduce_mean(tf.square(ys - output))
optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1150):
        sess.run(optimizer, feed_dict={xs: x_data, ys: y_data})
        if i % 10 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_realData}))

































