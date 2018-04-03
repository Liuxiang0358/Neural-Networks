import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time


def read_files(file_dir, n_data_set):
    # list initialization
    cat_data = []
    dog_data = []
    cat_label = []
    dog_label = []

    valid_format = [".jpg"]

    # read cat images
    for file in os.listdir(file_dir + "\\Cat"):
        ext = os.path.splitext(file)[1]
        if ext.lower() not in valid_format:
            continue
        cat_data.append(file_dir + "\\Cat\\" + file)
        cat_label.append(0)
        if len(cat_label) >= n_data_set:
            break

    # read dog images
    for file in os.listdir(file_dir + "\\Dog"):
        ext = os.path.splitext(file)[1]
        if ext.lower() not in valid_format:
            continue
        dog_data.append(file_dir + "\\Dog\\" + file)
        dog_label.append(1)
        if len(dog_label) >= n_data_set:
            break

    image_list = np.hstack((cat_data, dog_data))
    label_list = np.hstack((cat_label, dog_label))

    data_array = np.array([image_list, label_list]).transpose()
    np.random.shuffle(data_array)

    image_list = list(data_array[:, 0])
    label_list = list(data_array[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list



def get_batch(image, label, width, height, batch_size):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    train_input_queue = tf.train.slice_input_producer([image, label], shuffle=False)
    image_content = tf.read_file(train_input_queue[0])
    image = tf.image.decode_jpeg(image_content, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, target_width=width, target_height=height)
    image = tf.image.per_image_standardization(image)
    label = train_input_queue[1]
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=5000)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch



def inference (image, batch_size, n_class):
    with tf.variable_scope('conv_layer1') as scope:
        w = tf.get_variable('weights',
                            shape=[3, 3, 3, 16],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b = tf.get_variable('biases',
                            shape=[16],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
        conv1 = tf.nn.conv2d(image, w,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        conv1 = tf.nn.bias_add(conv1, b)
        output1 = tf.nn.relu(conv1, name=scope.name)

    with tf.variable_scope('pooling_layer1') as scope:
        pooling1 = tf.nn.max_pool(output1,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')

    with tf.variable_scope('conv_layer2') as scope:
        w2 = tf.get_variable('weights',
                            shape=[3, 3, 16, 16],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b2 = tf.get_variable('biases',
                            shape=[16],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(pooling1, w2,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        conv2 = tf.nn.bias_add(conv2, b2)
        output2 = tf.nn.relu(conv2)

    with tf.variable_scope('pooling_layer2') as scope:
        pooling2 = tf.nn.max_pool(output2,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')

    with tf.variable_scope('fully_connected1') as scope:
        output2_flat = tf.reshape(pooling2, shape=[batch_size, -1])
        dim = output2_flat.get_shape()[1].value
        w3 = tf.get_variable('weights',
                            shape=[dim, 128],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        b3 = tf.get_variable('biases',
                            shape=[128],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
        output3 = tf.nn.relu(tf.matmul(output2_flat, w3) + b3)

    with tf.variable_scope('fully_connected2') as scope:
        w4 = tf.get_variable('weights',
                            shape=[128, 128],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        b4 = tf.get_variable('biases',
                            shape=[128],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
        output4 = tf.nn.relu(tf.add(tf.matmul(output3, w4), b4))

    with tf.variable_scope('final_result') as scope:
        w5 = tf.get_variable('weights',
                            shape=[128, n_class],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        b5 = tf.get_variable('biases',
                            shape=[n_class],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
        output5 = tf.add(tf.matmul(output4, w5), b5)

    return output5

def loss_func(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    with tf.name_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, trainable=False)
        optimizer = optimizer.minimize(loss, global_step=global_step)
    return optimizer


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
    return accuracy



Batch_size = 10
Width = 150
Height = 150
n_data_set = 90
n_class = 2
max_step = 15000
learning_rate = 0.0001


def run_training():
    train_data_dir = "C:\\Users\\lyf58\\PycharmProjects\\CatVsDog\\catDog"

    image_list, label_list = read_files(train_data_dir, n_data_set)
    image_batch, label_batch = get_batch(image_list,
                                         label_list,
                                         Width,
                                         Height,
                                         Batch_size)

    train_logits = inference(image_batch, Batch_size, n_class)
    train_loss = loss_func(train_logits, label_batch)
    train_optimizer = training(train_loss, learning_rate)
    train_accuracy = evaluation(train_logits, label_batch)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for step in np.arange(max_step):
                if coord.should_stop():
                    break
                _, tra_loss, tra_acc = sess.run([train_optimizer, train_loss, train_accuracy])

                if step % 20 == 0:
                    print(step, tra_loss, tra_acc)

        except tf.errors.OutOfRangeError:
            print("error")
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


def gen_Frame(Frame_size):
    frame = np.zeros([Frame_size, Frame_size])
    ran_width = np.random.randint(1, Frame_size+1)
    ran_height = np.random.randint(1, Frame_size+1)
    ran_start_width = np.random.randint(0, Frame_size - ran_width + 1)
    ran_start_height = np.random.randint(0, Frame_size - ran_height + 1)
    filled_frame = frame.copy()
    frame[ran_start_width:ran_start_width + ran_width, ran_start_height] = 1
    frame[ran_start_width:ran_start_width + ran_width, ran_start_height + ran_height - 1] = 1
    frame[ran_start_width, ran_start_height:ran_start_height + ran_height] = 1
    frame[ran_start_width + ran_width - 1, ran_start_height:ran_start_height + ran_height] = 1
    for i in range(ran_start_width, ran_start_width + ran_width):
        for j in range(ran_start_height, ran_start_height + ran_height):
            filled_frame[i, j] = 1
    return frame, filled_frame

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
    w = tf.get_variable('weights',
                        shape=[3, 3, 1, 600],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b = tf.get_variable('biases',
                        shape=[600],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))
    conv1 = tf.nn.conv2d(frame, w,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b)
    conv1 = tf.nn.relu(conv1)

    # de_conv1 = tf.nn.bias_add(conv1, -b)
    de_conv1 = conv1

    de_conv1 = tf.nn.conv2d_transpose(de_conv1, w,
                                      output_shape = [1, frame_size ,frame_size ,1],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    output = tf.nn.relu(tf.add(de_conv1, tf.cast(frame, tf.float32)))

    return output


frame_size = 40
std_dtv = 0.1

# def training_model2(frame):

# _data, y_data = gen_noise_Frame(frame_size, 0.08)
# np.set_printoptions(precision=2)
# print(x_data,'\n',y_data)

x = tf.placeholder(tf.float32, [1, frame_size, frame_size, 1])
y = tf.placeholder(tf.float32, [1, frame_size, frame_size, 1])
output = training_model(x, frame_size)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output)))

optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss)
init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10000000):
        y_data, x_data = gen_noise_Frame(frame_size, std_dtv)
        x_data = x_data.reshape([1, frame_size, frame_size, 1])
        y_data = y_data.reshape([1, frame_size, frame_size, 1])
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})

        if epoch % 5000 == 0:
            image_show_filled = sess.run(output, feed_dict={x: x_data})
            image_show_filled = image_show_filled.reshape([frame_size, frame_size])
            image_show = x_data
            image_show = image_show.reshape([frame_size, frame_size])
            ground_truth = y_data.reshape([frame_size, frame_size])
            print(np.nanmean(((ground_truth - image_show) ** 2)), '\n', np.nanmean(((ground_truth - image_show_filled) ** 2)))
            # print(image_show, '\n', image_show_filled)
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(image_show, cmap='gray')
            axarr[1].imshow(image_show_filled, cmap='gray')
            plt.close()




























