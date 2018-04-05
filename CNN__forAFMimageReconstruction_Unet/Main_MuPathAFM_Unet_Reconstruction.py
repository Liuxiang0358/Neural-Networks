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


def get_batch_undersampled(image, width, height, batch_size, mupathLen, SamplingRatio):
    image = tf.cast(image, tf.string)
    mask = muPathMaskGen(mupathLen, width, height, SamplingRatio)
    mask_reverse = (mask == 0)
    mask = tf.cast(mask, tf.float32)
    mask_reverse = tf.cast(mask_reverse, tf.float32)

    train_input_queue = tf.train.slice_input_producer([image], shuffle=False)
    image_content = tf.read_file(train_input_queue[0])
    image = tf.image.decode_jpeg(image_content, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_image_with_crop_or_pad(image, target_width=width, target_height=height)
    image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image)
    undersampled_image = tf.multiply(tf.reshape(image, [width, height]), mask)
    undersampled_image = tf.reshape(undersampled_image, [width, height, 1])
    image_batch, undersampled_image_batch = tf.train.batch([image, undersampled_image], batch_size=batch_size)

    return undersampled_image_batch, image_batch, mask_reverse


def muPathMaskGen(mupathLen, Width, Height, SamplingRatio, Flag=False):
    pixelIfSampled = np.zeros([Width, Height])
    while np.sum(pixelIfSampled) < Width*Height*SamplingRatio:
        if Flag is True:
            i = np.random.randint(0, Width)
            j = np.random.randint(1 - mupathLen, Height)
            pixelIfSampled[i][np.maximum(j, 0):min(j + mupathLen, Width+1)] = 1
        else:
            i = np.random.randint(0, Width)
            j = np.random.randint(0, Height - mupathLen + 1)
            if np.sum(pixelIfSampled[i][j : j + mupathLen]) < 0.5:
                pixelIfSampled[i][j: j + mupathLen] = 1
    return pixelIfSampled






def CNN_U_net_deep4(frame, batch_size, width, height):

    w1 = tf.get_variable('weight1',
                        shape=[3, 3, 1, 64],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b1 = tf.get_variable('bias1',
                        shape=[64],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv1 = tf.nn.conv2d(frame, w1,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b1)
    conv1 = tf.nn.relu(conv1)


    w2 = tf.get_variable('weight2',
                        shape=[3, 3, 64, 64],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b2 = tf.get_variable('bias2',
                        shape=[64],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv2 = tf.nn.conv2d(conv1, w2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b2)
    conv2 = tf.nn.relu(conv2)


    pool1 = tf.nn.max_pool(conv2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')


    w3 = tf.get_variable('weight3',
                        shape=[3, 3, 64, 128],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b3 = tf.get_variable('bias3',
                        shape=[128],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv3 = tf.nn.conv2d(pool1, w3,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b3)
    conv3 = tf.nn.relu(conv3)


    pool2 = tf.nn.max_pool(conv3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    w4 = tf.get_variable('weight4',
                        shape=[3, 3, 128, 256],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b4 = tf.get_variable('bias4',
                        shape=[256],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv4 = tf.nn.conv2d(pool2, w4,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv4 = tf.nn.bias_add(conv4, b4)
    conv4 = tf.nn.relu(conv4)

    pool3 = tf.nn.max_pool(conv4,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    w5 = tf.get_variable('weight5',
                        shape=[3, 3, 256, 512],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b5 = tf.get_variable('bias5',
                        shape=[512],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv5 = tf.nn.conv2d(pool3, w5,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv5 = tf.nn.bias_add(conv5, b5)
    conv5 = tf.nn.relu(conv5)



    w6 = tf.get_variable('weight6',
                        shape=[3, 3, 512, 256],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b6 = tf.get_variable('bias6',
                        shape=[256],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv6 = tf.nn.conv2d(conv5, w6,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv6 = tf.nn.bias_add(conv6, b6)
    conv6 = tf.nn.relu(conv6)

    unpool1 = tf.image.resize_images(conv6, [int(width / 4), int(height / 4)])

    unpool1 = tf.concat([unpool1, conv4], axis=3)


    w7 = tf.get_variable('weight7',
                        shape=[3, 3, 512, 256],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b7 = tf.get_variable('bias7',
                        shape=[256],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv7 = tf.nn.conv2d(unpool1, w7,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv7 = tf.nn.bias_add(conv7, b7)
    conv7 = tf.nn.relu(conv7)


    w8 = tf.get_variable('weight8',
                        shape=[3, 3, 256, 128],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b8 = tf.get_variable('bias8',
                        shape=[128],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv8 = tf.nn.conv2d(conv7, w8,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv8 = tf.nn.bias_add(conv8, b8)
    conv8 = tf.nn.relu(conv8)


    unpool2 = tf.image.resize_images(conv8, [int(width / 2), int(height / 2)])

    unpool2 = tf.concat([unpool2, conv3], axis=3)


    w9 = tf.get_variable('weight9',
                        shape=[3, 3, 256, 128],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b9 = tf.get_variable('bias9',
                        shape=[128],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv9 = tf.nn.conv2d(unpool2, w9,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv9 = tf.nn.bias_add(conv9, b9)
    conv9 = tf.nn.relu(conv9)


    w10 = tf.get_variable('weight10',
                        shape=[3, 3, 128, 64],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b10 = tf.get_variable('bias10',
                        shape=[64],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv10 = tf.nn.conv2d(conv9, w10,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv10 = tf.nn.bias_add(conv10, b10)
    conv10 = tf.nn.relu(conv10)

    unpool3 = tf.image.resize_images(conv10, [int(width), int(height)])

    unpool3 = tf.concat([unpool3, conv2], axis=3)


    w11 = tf.get_variable('weight11',
                        shape=[3, 3, 128, 64],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b11 = tf.get_variable('bias11',
                        shape=[64],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv11 = tf.nn.conv2d(unpool3, w11,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv11 = tf.nn.bias_add(conv11, b11)
    conv11 = tf.nn.relu(conv11)


    w12 = tf.get_variable('weight12',
                        shape=[3, 3, 64, 64],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b12 = tf.get_variable('bias12',
                        shape=[64],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv12 = tf.nn.conv2d(conv11, w12,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv12 = tf.nn.bias_add(conv12, b12)
    conv12 = tf.nn.relu(conv12)


    w13 = tf.get_variable('weight13',
                        shape=[1, 1, 64, 1],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b13 = tf.get_variable('bias13',
                        shape=[1],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    output = tf.nn.conv2d(conv12, w13,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    output = tf.nn.bias_add(output, b13)
    output = tf.nn.relu(output)


    return output



def loss_func(y, output, mask):
    y = tf.reshape(y, mask.shape)
    output = tf.reshape(output, mask.shape)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.multiply(y - output, mask))))
    return loss

def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return optimizer



def run_training():
    train_data_dir = "D:\\myPythonProjects\\ImageDenoising\\CatVsDog"
    image_list = read_files(train_data_dir, n_data_set)
    undersampled_image_batch, image_batch, mask = get_batch_undersampled(image_list,
                                                             Width,
                                                             Height,
                                                             Batch_size,
                                                             mupathLen,
                                                             SamplingRatio)

    output = CNN_U_net_deep4(undersampled_image_batch, Batch_size, Width, Height)
    train_loss = loss_func(output, image_batch, mask)
    train_optimizer = training(train_loss, learning_rate)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for step in np.arange(max_step):
                if coord.should_stop():
                    break
                _, tra_loss = sess.run([train_optimizer, train_loss])

                if step % 20 == 0:
                    print(step, tra_loss)

                if (step+1) % 100 == 0:
                    save_path = saver.save(sess, "D:\\myPythonProjects\\ModelParameters\\model_reconstruction_mu35_ptg20_Unet.ckpt")
                    print("Save to path:", save_path)

                    undersampled_image, img, recovered_image = sess.run([undersampled_image_batch, image_batch, output])
                    f, axarr = plt.subplots(1, 3)
                    axarr[0].imshow(undersampled_image[0, :, :, 0], cmap='gray')
                    axarr[0].set_title("undersampled image")
                    axarr[1].imshow(recovered_image[0, :, :, 0], cmap='gray')
                    axarr[1].set_title("recovered image")
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



Batch_size = 1
Width = 256
Height = 256
n_data_set = 10000
max_step = 1500000
learning_rate = 0.0001
mupathLen = 35
SamplingRatio = 0.2



run_training()

















