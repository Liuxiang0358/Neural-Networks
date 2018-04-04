import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os



def read_files(file_dir, n_data_set):
    # list initialization
    cat_data = []
    dog_data = []

    valid_format = [".jpg"]

    # read cat images
    for file in os.listdir(file_dir + "\\Cat"):
        ext = os.path.splitext(file)[1]
        if ext.lower() not in valid_format:
            continue
        cat_data.append(file_dir + "\\Cat\\" + file)
        if len(cat_data) >= n_data_set:
            break

    # read dog images
    for file in os.listdir(file_dir + "\\Dog"):
        ext = os.path.splitext(file)[1]
        if ext.lower() not in valid_format:
            continue
        dog_data.append(file_dir + "\\Dog\\" + file)
        if len(dog_data) >= n_data_set:
            break

    image_list = np.hstack((cat_data, dog_data))
    data_array = np.array([image_list]).transpose()
    np.random.shuffle(data_array)

    image_list = list(data_array[:, 0])

    return image_list


def get_batch(image, width, height, batch_size, stddev):
    image = tf.cast(image, tf.string)

    train_input_queue = tf.train.slice_input_producer([image], shuffle=False)
    image_content = tf.read_file(train_input_queue[0])
    image = tf.image.decode_jpeg(image_content, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_image_with_crop_or_pad(image, target_width=width, target_height=height)
    image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image)
    noised_image_batch = image
    noised_image_batch += tf.random_normal(shape=tf.shape(image), mean=0., stddev=stddev, dtype=tf.float32)
    image_batch, noised_image_batch = tf.train.batch([image, noised_image_batch], batch_size=batch_size)

    return noised_image_batch, image_batch



def CNN_symmetricSkip (frame, batch_size, width, height):

    w1 = tf.get_variable('weights',
                        shape=[3, 3, 1, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b1 = tf.get_variable('biases',
                        shape=[200],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv1 = tf.nn.conv2d(frame, w1,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b1)


    w2 = tf.get_variable('weights2',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b2 = tf.get_variable('biases2',
                        shape=[200],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv2 = tf.nn.conv2d(conv1, w2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b2)


    w3 = tf.get_variable('weights3',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b3 = tf.get_variable('biases3',
                        shape=[200],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv3 = tf.nn.conv2d(conv2, w3,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b3)


    w4 = tf.get_variable('weights4',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b4 = tf.get_variable('biases4',
                        shape=[200],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv4 = tf.nn.conv2d(conv3, w4,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv4 = tf.nn.bias_add(conv4, b4)


    w5 = tf.get_variable('weights5',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b5 = tf.get_variable('biases5',
                        shape=[200],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv5 = tf.nn.conv2d(conv4, w5,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv5 = tf.nn.bias_add(conv5, b5)


    conv5 = tf.nn.relu(conv5)


    d_w5 = tf.get_variable('dweights5',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))

    de_conv5 = tf.nn.conv2d_transpose(conv5, d_w5,
                                      output_shape = [batch_size, width, height, 200],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    de_conv5 = tf.nn.relu(tf.add(de_conv5, conv4))



    d_w4 = tf.get_variable('dweights4',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))


    de_conv4 = tf.nn.conv2d_transpose(de_conv5, d_w4,
                                      output_shape = [batch_size, width, height, 200],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    de_conv4 = tf.nn.relu(tf.add(de_conv4, conv3))


    d_w3 = tf.get_variable('dweights3',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))


    de_conv3 = tf.nn.conv2d_transpose(de_conv4, d_w3,
                                      output_shape = [batch_size, width, height, 200],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    de_conv3 = tf.nn.relu(tf.add(de_conv3, conv2))


    d_w2 = tf.get_variable('dweights2',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))


    de_conv2 = tf.nn.conv2d_transpose(de_conv3, d_w2,
                                      output_shape = [batch_size, width, height, 200],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    de_conv2 = tf.nn.relu(tf.add(de_conv2, conv1))


    d_w1 = tf.get_variable('dweights1',
                        shape=[3, 3, 1, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))


    de_conv1 = tf.nn.conv2d_transpose(de_conv2, d_w1,
                                      output_shape = [batch_size, width, height, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    output = tf.nn.relu(tf.add(de_conv1, tf.cast(frame, tf.float32)))


    return output





def CNN_symmetricSkip_8L (frame, batch_size, width, height):
    w1 = tf.get_variable('weights',
                         shape=[3, 3, 1, 250],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b1 = tf.get_variable('biases',
                         shape=[250],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    conv1 = tf.nn.conv2d(frame, w1,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b1)

    w2 = tf.get_variable('weights2',
                         shape=[3, 3, 250, 250],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b2 = tf.get_variable('biases2',
                         shape=[250],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    conv2 = tf.nn.conv2d(conv1, w2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b2)

    w3 = tf.get_variable('weights3',
                         shape=[3, 3, 250, 250],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b3 = tf.get_variable('biases3',
                         shape=[250],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    conv3 = tf.nn.conv2d(conv2, w3,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b3)

    w4 = tf.get_variable('weights4',
                         shape=[3, 3, 250, 250],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b4 = tf.get_variable('biases4',
                         shape=[250],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    conv4 = tf.nn.conv2d(conv3, w4,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv4 = tf.nn.bias_add(conv4, b4)

    conv4 = tf.nn.relu(conv4)

    d_w4 = tf.get_variable('dweights4',
                           shape=[3, 3, 250, 250],
                           dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))

    de_conv4 = tf.nn.conv2d_transpose(conv4, d_w4,
                                      output_shape=[batch_size, width, height, 250],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    de_conv4 = tf.nn.relu(tf.add(de_conv4, conv3))

    d_w3 = tf.get_variable('dweights3',
                           shape=[3, 3, 250, 250],
                           dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))

    de_conv3 = tf.nn.conv2d_transpose(de_conv4, d_w3,
                                      output_shape=[batch_size, width, height, 250],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    de_conv3 = tf.nn.relu(tf.add(de_conv3, conv2))

    d_w2 = tf.get_variable('dweights2',
                           shape=[3, 3, 250, 250],
                           dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))

    de_conv2 = tf.nn.conv2d_transpose(de_conv3, d_w2,
                                      output_shape=[batch_size, width, height, 250],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    de_conv2 = tf.nn.relu(tf.add(de_conv2, conv1))

    d_w1 = tf.get_variable('dweights1',
                           shape=[3, 3, 1, 250],
                           dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))

    de_conv1 = tf.nn.conv2d_transpose(de_conv2, d_w1,
                                      output_shape=[batch_size, width, height, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    output = tf.nn.relu(tf.add(de_conv1, tf.cast(frame, tf.float32)))

    return output




def CNN_symmetricSkip_6L (frame, batch_size, width, height):

    w1 = tf.get_variable('weights',
                        shape=[3, 3, 1, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b1 = tf.get_variable('biases',
                        shape=[200],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))
    conv1 = tf.nn.conv2d(frame, w1,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b1)


    w2 = tf.get_variable('weights2',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b2 = tf.get_variable('biases2',
                        shape=[200],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))
    conv2 = tf.nn.conv2d(conv1, w2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b2)


    w3 = tf.get_variable('weights3',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))
    b3 = tf.get_variable('biases3',
                        shape=[200],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))
    conv3 = tf.nn.conv2d(conv2, w3,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b3)

    conv3 = tf.nn.relu(conv3)


    d_w3 = tf.get_variable('dweights3',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))

    de_conv3 = tf.nn.conv2d_transpose(conv3, d_w3,
                                      output_shape = [batch_size, width, height, 200],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    de_conv3 = tf.nn.relu(tf.add(de_conv3, conv2))

    d_w2 = tf.get_variable('dweights2',
                        shape=[3, 3, 200, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))


    de_conv2 = tf.nn.conv2d_transpose(de_conv3, d_w2,
                                      output_shape = [batch_size, width, height, 200],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    de_conv2 = tf.nn.relu(tf.add(de_conv2, conv1))

    d_w1 = tf.get_variable('dweight1',
                        shape=[3, 3, 1, 200],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.003, dtype=tf.float32))

    de_conv1 = tf.nn.conv2d_transpose(de_conv2, d_w1,
                                      output_shape = [batch_size, width, height, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    output = tf.nn.relu(tf.add(de_conv1, tf.cast(frame, tf.float32)))


    return output




def CNN_symmetricSkip_4L(frame, batch_size, width, height):
    w1 = tf.get_variable('weights',
                        shape=[3, 3, 1, 900],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b1 = tf.get_variable('biases',
                        shape=[900],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))
    conv1 = tf.nn.conv2d(frame, w1,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b1)


    w2 = tf.get_variable('weights2',
                        shape=[3, 3, 900, 900],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b2 = tf.get_variable('biases2',
                        shape=[900],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))
    conv2 = tf.nn.conv2d(conv1, w2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b2)

    conv2 = tf.nn.relu(conv2)

    de_conv2 = tf.nn.conv2d_transpose(conv2, w2,
                                      output_shape = [batch_size, width, height, 900],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    de_conv2 = tf.nn.relu(tf.add(de_conv2, conv1))

    de_conv1 = tf.nn.conv2d_transpose(de_conv2, w1,
                                      output_shape = [batch_size, width, height, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

    output = tf.nn.relu(tf.add(de_conv1, tf.cast(frame, tf.float32)))

    return output




def loss_func(y, output):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output)))
    return loss

def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    #global_step = tf.Variable(0, trainable=False)
    #optimizer = optimizer.minimize(loss, global_step=global_step)
    return optimizer



Batch_size = 1
Width = 180
Height = 180
n_data_set = 4000
max_step = 15000
learning_rate = 0.00002
stddev = 12


def show_image():
    train_data_dir = "D:\\myPythonProjects\\ImageDenoising\\CatVsDog"
    image_list = read_files(train_data_dir, n_data_set)
    noised_image_batch, image_batch = get_batch(image_list,
                                                 Width,
                                                 Height,
                                                 Batch_size,
                                                 stddev)
    output = CNN_symmetricSkip_8L(noised_image_batch, Batch_size, Width, Height)
    train_loss = loss_func(output, image_batch)
    train_optimizer = training(train_loss, learning_rate)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, "D:\\myPythonProjects\\ModelParameters\\model_denoising_Skip8Layer.ckpt")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for step in np.arange(max_step):
                if coord.should_stop():
                    break
                nosied_img, img, denoised_image = sess.run([noised_image_batch, image_batch, output])
                f, axarr = plt.subplots(1, 3)
                axarr[0].imshow(nosied_img[0, :, :, 0], cmap='gray')
                axarr[0].set_title("noised image")
                axarr[1].imshow(denoised_image[0, :, :, 0], cmap='gray')
                axarr[1].set_title("denoised image")
                axarr[2].imshow(img[0, :, :, 0], cmap='gray')
                axarr[2].set_title("image")
                plt.show()
                plt.close()

        except tf.errors.OutOfRangeError:
            print("error")
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()



show_image()


























































