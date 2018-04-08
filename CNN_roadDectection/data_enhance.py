import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import scipy.misc
import imutils


def read_files(file_dir):
    valid_format = [".png"]

    image = []

    for file in os.listdir(file_dir):
        ext = os.path.splitext(file)[1]
        if ext.lower() not in valid_format:
            continue
        image.append(file_dir + "\\" + file)


    return image


file_dir = "C:\\Users\\lyf58\\PycharmProjects\\ImageDenoising\\roadData\\training\\image_2"
mask_dir = "C:\\Users\\lyf58\\PycharmProjects\\ImageDenoising\\roadData\\training\\gt_image_2"

file_dir_save = "C:\\Users\\lyf58\\PycharmProjects\\ImageDenoising\\roadData\\training\\data_enhanced\\"
mask_dir_save = "C:\\Users\\lyf58\\PycharmProjects\\ImageDenoising\\roadData\\training\\label_enhanced\\"

image = read_files(file_dir)
label = read_files(mask_dir)

cnt_image = 0
cnt_label = 0

fix_size_m = 1242
fix_size_n = 375

for s in range(len(image)):

    img = cv.imread(image[s], 3)
    lbl = cv.imread(label[s], 0)
    n, m = np.shape(lbl)

    binary_label = (lbl == 105 * np.ones((n, m))).astype(float)

    # save original images
    scipy.misc.imsave(file_dir_save + str(cnt_image) + ".png", cv.resize(img, (fix_size_m, fix_size_n)))
    cnt_image += 1
    binary_label = np.round(cv.resize(binary_label, (fix_size_m, fix_size_n)))
    binary_label.astype(int)
    scipy.misc.imsave(mask_dir_save + str(cnt_label) + ".png", binary_label)
    cnt_label += 1

    # save flip images
    scipy.misc.imsave(file_dir_save + str(cnt_image) + ".png", cv.flip(cv.resize(img, (fix_size_m, fix_size_n)), 1))
    cnt_image += 1
    scipy.misc.imsave(mask_dir_save + str(cnt_label) + ".png", cv.flip(binary_label, 1))
    cnt_label += 1

# random cut
for s in range(len(image)):

    img = cv.imread(image[s], 3)
    lbl = cv.imread(label[s], 0)
    img = cv.resize(img, (fix_size_m, fix_size_n))
    n, m = np.shape(lbl)
    lbl = (lbl == 105 * np.ones((n, m))).astype(float)
    lbl = np.round(cv.resize(lbl, (fix_size_m, fix_size_n)))
    n, m = np.shape(lbl)

    for i in range(10):
        rand_left = np.random.randint(0, round(0.2*m))
        rand_right = np.random.randint(0, round(0.2 * m))
        rand_top = np.random.randint(0, round(0.18 * n))
        rand_bottom = np.random.randint(0, round(0.18 * n))
        new_img = img[rand_top:n - rand_top - rand_bottom, rand_left:m - rand_left - rand_right]
        new_lbl = lbl[rand_top:n - rand_top - rand_bottom, rand_left:m - rand_left - rand_right]
        new_img = cv.resize(new_img, (m, n))
        new_lbl = np.round(cv.resize(new_lbl, (m, n)))
        new_lbl = new_lbl.astype(int)

        scipy.misc.imsave(file_dir_save + str(cnt_image) + ".png", new_img)
        cnt_image += 1
        scipy.misc.imsave(mask_dir_save + str(cnt_label) + ".png", new_lbl)
        cnt_label += 1



# flip random cut
for s in range(len(image)):

    img = cv.imread(image[s], 3)
    img = cv.flip(img, 1)
    lbl = cv.imread(label[s], 0)
    lbl = cv.flip(lbl, 1)
    img = cv.resize(img, (fix_size_m, fix_size_n))
    n, m = np.shape(lbl)
    lbl = (lbl == 105 * np.ones((n, m))).astype(float)
    lbl = np.round(cv.resize(lbl, (fix_size_m, fix_size_n)))
    n, m = np.shape(lbl)

    for i in range(10):
        rand_left = np.random.randint(0, round(0.2 * m))
        rand_right = np.random.randint(0, round(0.2 * m))
        rand_top = np.random.randint(0, round(0.18 * n))
        rand_bottom = np.random.randint(0, round(0.18 * n))
        new_img = img[rand_top:n - rand_top - rand_bottom, rand_left:m - rand_left - rand_right]
        new_lbl = lbl[rand_top:n - rand_top - rand_bottom, rand_left:m - rand_left - rand_right]
        new_img = cv.resize(new_img, (m, n))
        new_lbl = np.round(cv.resize(new_lbl, (m, n)))
        new_lbl = new_lbl.astype(int)

        scipy.misc.imsave(file_dir_save + str(cnt_image) + ".png", new_img)
        cnt_image += 1
        scipy.misc.imsave(mask_dir_save + str(cnt_label) + ".png", new_lbl)
        cnt_label += 1



image = read_files(file_dir_save)
label = read_files(mask_dir_save)

for s in range(len(image)):
    img = cv.imread(image[s], 3)
    lbl = cv.imread(label[s], 0)
    lbl.astype(float)
    n, m = np.shape(lbl)

    for angle in [2.5,3.5,4.5,5.5]:

        if (angle == 2.5):
            vertical_cut = 7
            side_cut = 33
        elif (angle == 3.5):
            vertical_cut = 8
            side_cut = 38
        elif (angle == 4.5):
            vertical_cut = 12
            side_cut = 48
        else:
            vertical_cut = 16
            side_cut = 58

            new_img = imutils.rotate(img, angle)
            new_lbl = imutils.rotate(lbl, angle)
            new_img = new_img[side_cut:n-side_cut,vertical_cut:m-vertical_cut]
            new_lbl = new_lbl[side_cut:n-side_cut,vertical_cut:m-vertical_cut]

            new_img = cv.resize(new_img, (m, n))
            new_lbl = cv.resize(new_lbl, (m, n))

            scipy.misc.imsave(file_dir_save + str(cnt_image) + ".png", new_img)
            cnt_image += 1
            new_lbl = np.round(new_lbl).astype(int)
            scipy.misc.imsave(mask_dir_save + str(cnt_label) + ".png", new_lbl)
            cnt_label += 1

 





















