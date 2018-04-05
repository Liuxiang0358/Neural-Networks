import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def read_files(file_dir, n_data_set):
    # list initialization
    cat_data = []
    dog_data = []

    valid_format = [".jpg"]

    # read Lena images
    for file in os.listdir(file_dir + "\\Lena"):
        ext = os.path.splitext(file)[1]
        if ext.lower() not in valid_format:
            continue
        cat_data.append(file_dir + "\\Lena\\" + file)
        if len(cat_data) >= n_data_set:
            break


    # read cat images
    # for file in os.listdir(file_dir + "\\Cat"):
    #     ext = os.path.splitext(file)[1]
    #     if ext.lower() not in valid_format:
    #         continue
    #     cat_data.append(file_dir + "\\Cat\\" + file)
    #     if len(cat_data) >= n_data_set:
    #         break

    # read dog images
    # for file in os.listdir(file_dir + "\\Dog"):
    #     ext = os.path.splitext(file)[1]
    #     if ext.lower() not in valid_format:
    #         continue
    #     dog_data.append(file_dir + "\\Dog\\" + file)
    #     if len(dog_data) >= n_data_set:
    #         break

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

    train_input_queue = tf.train.slice_input_producer([image], shuffle=True)
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



def CNN_symmetricSkip_8L (frame, batch_size, width, height):

    w1 = tf.get_variable('weights1',
                        shape=[3, 3, 1, 290],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b1 = tf.get_variable('biases1',
                        shape=[290],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv1 = tf.nn.conv2d(frame, w1,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b1)

    conv1 = tf.nn.relu(conv1)



    w2 = tf.get_variable('weights2',
                        shape=[3, 3, 290, 290],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b2 = tf.get_variable('biases2',
                        shape=[290],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv2 = tf.nn.conv2d(conv1, w2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b2)

    conv2 = tf.nn.relu(conv2)



    w3 = tf.get_variable('weights3',
                        shape=[3, 3, 290, 290],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b3 = tf.get_variable('biases3',
                        shape=[290],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv3 = tf.nn.conv2d(conv2, w3,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b3)

    conv3 = tf.nn.relu(conv3)


    w4 = tf.get_variable('weights4',
                        shape=[3, 3, 290, 290],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b4 = tf.get_variable('biases4',
                        shape=[290],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv4 = tf.nn.conv2d(conv3, w4,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv4 = tf.nn.bias_add(conv4, b4)

    conv4 = tf.nn.relu(conv4)


    w5 = tf.get_variable('weights5',
                        shape=[3, 3, 290, 290],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b5 = tf.get_variable('biases5',
                        shape=[290],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv5 = tf.nn.conv2d(conv4, w5,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv5 = tf.nn.bias_add(conv5, b5)

    conv5 = tf.nn.relu(conv5)


    w6 = tf.get_variable('weights6',
                        shape=[3, 3, 290, 290],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b6 = tf.get_variable('biases6',
                        shape=[290],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv6 = tf.nn.conv2d(conv5, w6,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv6 = tf.nn.bias_add(conv6, b6)

    conv6 = tf.nn.relu(conv6)


    w7 = tf.get_variable('weights7',
                        shape=[3, 3, 290, 290],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b7 = tf.get_variable('biases7',
                        shape=[290],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv7 = tf.nn.conv2d(conv6, w7,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv7 = tf.nn.bias_add(conv7, b7)

    conv7 = tf.nn.relu(conv7)


    w8 = tf.get_variable('weights8',
                        shape=[3, 3, 290, 1],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b8 = tf.get_variable('biases8',
                        shape=[1],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    output = tf.nn.conv2d(conv7, w8,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    output = tf.nn.bias_add(output, b8)

    output = tf.nn.relu(output)

    return output



def image_show():
    train_data_dir = "C:\\Users\\lyf58\\PycharmProjects\\ImageDenoising\\CatVsDog"
    image_list = read_files(train_data_dir, n_data_set)
    undersampled_image_batch, image_batch, mask = get_batch_undersampled(image_list,
                                                             Width,
                                                             Height,
                                                             Batch_size,
                                                             mupathLen,
                                                             SamplingRatio)

    output = CNN_symmetricSkip_8L(undersampled_image_batch, Batch_size, Width, Height)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, "C:\\Users\\lyf58\\PycharmProjects\\ModelParameters\\model_reconstruction_mu35_ptg20_fullyConvolution.ckpt")


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for _ in np.arange(max_step):
                if coord.should_stop():
                    break
                undersampled_image, img, mask_reverse, recovered_image = sess.run([undersampled_image_batch, image_batch, mask, output])
                mask_array = (mask_reverse == 0).astype(float)
                ax1 = plt.subplot(1, 3, 1)
                ax1.imshow(undersampled_image[0, :, :, 0], cmap='gray')
                ax1.set_title("Mu-path undersampled image")
                ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
                ax2.imshow(np.multiply(recovered_image[0, :, :, 0], mask_reverse) + np.multiply(img[0, :, :, 0], mask_array), cmap='gray')
                ax2.set_title("recovered image")
                ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
                ax3.imshow(img[0, :, :, 0], cmap='gray')
                ax3.set_title("image")
                plt.show()
                plt.close()


        except tf.errors.OutOfRangeError:
            print("error")
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()



Batch_size = 1
Width = 290
Height = 290
n_data_set = 20
max_step = 150000
mupathLen = 35
SamplingRatio = 0.2



image_show()

















