import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



def gen_noise_Frame(Frame_size, std_dtv):
    frame = np.zeros([Frame_size, Frame_size])
    ran_width = np.random.randint(1, Frame_size+1)
    ran_height = np.random.randint(1, Frame_size+1)
    ran_start_width = np.random.randint(0, Frame_size - ran_width + 1)
    ran_start_height = np.random.randint(0, Frame_size - ran_height + 1)
    frame[ran_start_width:ran_start_width + ran_width, ran_start_height] = 1
    frame[ran_start_width:ran_start_width + ran_width, ran_start_height + ran_height - 1] = 1
    frame[ran_start_width, ran_start_height:ran_start_height + ran_height] = 1
    frame[ran_start_width + ran_width - 1, ran_start_height:ran_start_height + ran_height] = 1
    noise_frame = frame.copy()
    noise_frame += np.random.normal(loc=0.0, scale=std_dtv, size=[Frame_size, Frame_size])
    return frame, noise_frame


def training_model(frame, frame_size):
    w1 = tf.get_variable('weights',
                        shape=[3, 3, 1, 150],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b1 = tf.get_variable('biases',
                        shape=[150],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv1 = tf.nn.conv2d(frame, w1,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b1)


    w2 = tf.get_variable('weights2',
                        shape=[3, 3, 150, 150],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b2 = tf.get_variable('biases2',
                        shape=[150],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv2 = tf.nn.conv2d(conv1, w2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b2)

    conv2 = tf.nn.relu(conv2)

    de_conv2 = tf.nn.conv2d_transpose(conv2, w2,
                                      output_shape = [1, frame_size, frame_size, 150],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    de_conv2 = tf.nn.relu(tf.add(de_conv2, conv1))

    de_conv1 = tf.nn.conv2d_transpose(de_conv2, w1,
                                      output_shape = [1, frame_size, frame_size, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    output = tf.nn.relu(tf.add(de_conv1, tf.cast(frame, tf.float32)))

    return output





frame_size = 30
std_dtv = 0.1

x = tf.placeholder(tf.float32, [1, frame_size, frame_size, 1])
y = tf.placeholder(tf.float32, [1, frame_size, frame_size, 1])
output = training_model(x, frame_size)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output)))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
init = tf.global_variables_initializer()




with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10000000):
        y_data, x_data = gen_noise_Frame(frame_size, std_dtv)
        x_data = x_data.reshape([1, frame_size, frame_size, 1])
        y_data = y_data.reshape([1, frame_size, frame_size, 1])
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})

        if epoch % 14000 == 0:
            image_show_filled = sess.run(output, feed_dict={x: x_data})
            image_show_filled = image_show_filled.reshape([frame_size, frame_size])
            image_show = x_data
            image_show = image_show.reshape([frame_size, frame_size])
            ground_truth = y_data.reshape([frame_size, frame_size])
            print(np.nanmean(((ground_truth - image_show) ** 2)), '\n', np.nanmean(((ground_truth - image_show_filled) ** 2)))
            # print(image_show, '\n', image_show_filled)
            f, axarr = plt.subplots(1, 3)
            axarr[0].imshow(image_show, cmap='gray')
            axarr[0].set_title("Noised image")
            axarr[1].imshow(image_show_filled, cmap='gray')
            axarr[1].set_title("Denoised image by Deconvolution Neural Network")
            axarr[2].imshow(ground_truth, cmap='gray')
            axarr[2].set_title("Ground Truth")
            plt.close()










