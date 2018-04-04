import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def gen_Frame(Frame_size):
    frame = np.zeros([Frame_size, Frame_size])
    ran_width = np.random.randint(1, Frame_size+1)
    ran_height = np.random.randint(1, Frame_size+1)
    ran_start_width = np.random.randint(0, Frame_size - ran_width + 1)
    ran_start_height = np.random.randint(0, Frame_size - ran_height + 1)
    frame[ran_start_width:ran_start_width + ran_width, ran_start_height] = 1
    frame[ran_start_width:ran_start_width + ran_width, ran_start_height + ran_height - 1] = 1
    frame[ran_start_width, ran_start_height:ran_start_height + ran_height] = 1
    frame[ran_start_width + ran_width - 1, ran_start_height:ran_start_height + ran_height] = 1
    filled_frame = frame.copy()
    for i in range(ran_start_width + 1, ran_start_width + ran_width):
        for j in range(ran_start_height + 1, ran_start_height + ran_height):
            filled_frame[i][j] = 1
    return frame, filled_frame



def training_Unet_model(frame, frame_size):
    w1 = tf.get_variable('w1',
                        shape=[3, 3, 1, 64],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b1 = tf.get_variable('b1',
                        shape=[64],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv1 = tf.nn.conv2d(frame, w1,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b1)
    conv1 = tf.nn.relu(conv1)

    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    w2 = tf.get_variable('w2',
                        shape=[3, 3, 64, 128],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b2 = tf.get_variable('b2',
                        shape=[128],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv2 = tf.nn.conv2d(pool1, w2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b2)
    conv2 = tf.nn.relu(conv2)

    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    w3 = tf.get_variable('w3',
                        shape=[3, 3, 128, 256],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b3 = tf.get_variable('b3',
                        shape=[256],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv3 = tf.nn.conv2d(pool2, w3,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b3)
    conv3 = tf.nn.relu(conv3)

    w4 = tf.get_variable('w4',
                        shape=[3, 3, 256, 128],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b4 = tf.get_variable('b4',
                        shape=[128],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv4 = tf.nn.conv2d(conv3, w4,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv4 = tf.nn.bias_add(conv4, b4)
    conv4 = tf.nn.relu(conv4)

    unpool1 = tf.image.resize_images(conv4, [int(frame_size/2), int(frame_size/2)])

    unpool1 = tf.concat([unpool1, conv2], axis=3)

    w5 = tf.get_variable('w5',
                        shape=[3, 3, 256, 128],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b5 = tf.get_variable('b5',
                        shape=[128],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv5 = tf.nn.conv2d(unpool1, w5,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv5 = tf.nn.bias_add(conv5, b5)
    conv5 = tf.nn.relu(conv5)

    w6 = tf.get_variable('w6',
                        shape=[3, 3, 128, 64],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b6 = tf.get_variable('b6',
                        shape=[64],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv6 = tf.nn.conv2d(conv5, w6,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv6 = tf.nn.bias_add(conv6, b6)
    conv6 = tf.nn.relu(conv6)


    unpool2 = tf.image.resize_images(conv6, [int(frame_size), int(frame_size)])
    unpool2 = tf.concat([unpool2, conv1], axis=3)


    w7 = tf.get_variable('w7',
                        shape=[3, 3, 128, 64],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b7 = tf.get_variable('b7',
                        shape=[64],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv7 = tf.nn.conv2d(unpool2, w7,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv7 = tf.nn.bias_add(conv7, b7)
    conv7 = tf.nn.relu(conv7)

    w8 = tf.get_variable('w8',
                        shape=[1, 1, 64, 1],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b8 = tf.get_variable('b8',
                        shape=[1],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv8 = tf.nn.conv2d(conv7, w8,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv8 = tf.nn.bias_add(conv8, b8)
    conv8 = tf.nn.relu(conv8)


    return conv8





frame_size = 60

x = tf.placeholder(tf.float32, [1, frame_size, frame_size, 1])
y = tf.placeholder(tf.float32, [1, frame_size, frame_size, 1])
output = training_Unet_model(x, frame_size)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output)))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10000000):
        x_data, y_data = gen_Frame(frame_size)
        x_data = x_data.reshape([1, frame_size, frame_size, 1])
        y_data = y_data.reshape([1, frame_size, frame_size, 1])
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})

        if epoch % 3000 == 0:
            image_show_filled = sess.run(output, feed_dict={x: x_data})
            image_show_filled = image_show_filled
            image_show_filled = image_show_filled.reshape([frame_size, frame_size])
            x_data = x_data
            x_data = x_data.reshape([frame_size, frame_size])
            y_data = y_data
            ground_truth = y_data.reshape([frame_size, frame_size])
            print(np.nanmean(((ground_truth - x_data) ** 2)), '\n', np.nanmean(((ground_truth - image_show_filled) ** 2)))
            f, axarr = plt.subplots(1, 3)
            axarr[0].imshow(x_data, cmap='gray')
            axarr[0].set_title("frame image")
            axarr[1].imshow(image_show_filled, cmap='gray')
            axarr[1].set_title("Filled image by U Neural Network")
            axarr[2].imshow(ground_truth, cmap='gray')
            axarr[2].set_title("Ground Truth")
            plt.close()














