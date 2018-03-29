import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x_data = np.linspace(-0.5, 0.5, 400)[:, np.newaxis]

noise = np.random.normal(0, 0.01, x_data.shape)

y_data = np.power(x_data, 3) + noise


x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])


weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))

Wx_plus_b_l1 = tf.matmul(x, weights_L1) + biases_L1

L1 = tf.nn.tanh(Wx_plus_b_l1)

weights_output = tf.Variable(tf.random_normal([10, 1]))
biases_output = tf.Variable(tf.zeros([1,1]))
output = tf.matmul(L1, weights_output) + biases_output

prediction = tf.nn.tanh(output)

loss = tf.reduce_mean(tf.square(y - prediction))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(optimizer,feed_dict={x:x_data, y:y_data})

    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value,'r-',lw=5)
    plt.show()




































