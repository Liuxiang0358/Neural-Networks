
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import cv2
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle






MIDLEARNED3 = np.load('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED3\\reward.npy')
MIDLEARNED2 = np.load('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED2\\reward.npy')

MIDLEARNED1 = np.load('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED1\\reward.npy')
len1 = len(MIDLEARNED1)-1

MIDLEARNED5 = np.load('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED5\\reward.npy')
len2 = len(MIDLEARNED5)


for k in range(128):

    img3 = mpimg.imread('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED3\\' + str(k+1) + '.jpg')
    img8 = mpimg.imread('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED3\\NN' + str(k + 1) + '.jpg')

    img2 = mpimg.imread('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED2\\' + str(k+1) + '.jpg')
    img7 = mpimg.imread('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED2\\NN' + str(k + 1) + '.jpg')

    img1 = mpimg.imread('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED1\\' + str(np.min((k+1, len1))) + '.jpg')
    img6 = mpimg.imread('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED1\\NN' + str(np.min((k+1, len1))) + '.jpg')

    img5 = mpimg.imread('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED5\\' + str(np.min((k + 1, len2))) + '.jpg')
    img10 = mpimg.imread('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED5\\NN' + str(np.min((k + 1, len2))) + '.jpg')

    fig, axes = plt.subplots(2, 5)
    # fig.tight_layout()
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)

    ax1 = plt.subplot(251)
    ax1.set_title("First attempt", fontsize=8)
    plt.imshow(img1)
    ax1.set_xlabel('Cum. discounted reward:\n' + str(np.round(MIDLEARNED1[np.min((k,len1-1))]/10,3)), fontsize=6)
    plt.xticks([])
    plt.yticks([])

    ax6 = plt.subplot(256)
    plt.imshow(img6)
    ax6.set_xlabel('Neural Network Estimation', fontsize=4)
    plt.xticks([])
    plt.yticks([])
    ax6.spines['bottom'].set_color('white')
    ax6.spines['top'].set_color('white')
    ax6.spines['right'].set_color('white')
    ax6.spines['left'].set_color('white')




    ax2 = plt.subplot(252)
    ax2.set_title("After 20 attempts", fontsize=8)
    plt.imshow(img2)
    # plt.axis('off')
    ax2.set_xlabel('Cum. discounted reward:\n' + str(np.round(MIDLEARNED2[k]/10,3)), fontsize=6)
    plt.xticks([])
    plt.yticks([])


    ax7 = plt.subplot(257)
    plt.imshow(img7)
    ax7.set_xlabel('Neural Network Estimation', fontsize=4)
    plt.xticks([])
    plt.yticks([])
    ax7.spines['bottom'].set_color('white')
    ax7.spines['top'].set_color('white')
    ax7.spines['right'].set_color('white')
    ax7.spines['left'].set_color('white')




    ax3 = plt.subplot(253)
    ax3.set_title("After 60 attempts", fontsize=8)
    plt.imshow(img3)
    ax3.set_xlabel('Cum. discounted reward:\n' + str(np.round(MIDLEARNED3[k]/10,3)), fontsize=6)
    plt.xticks([])
    plt.yticks([])


    ax8 = plt.subplot(258)
    plt.imshow(img8)
    ax8.set_xlabel('Neural Network Estimation', fontsize=4)
    plt.xticks([])
    plt.yticks([])
    ax8.spines['bottom'].set_color('white')
    ax8.spines['top'].set_color('white')
    ax8.spines['right'].set_color('white')
    ax8.spines['left'].set_color('white')



    ax4 = plt.subplot(254)
    ax4.set_title("More boxes added", fontsize=8)
    plt.imshow(img3)
    ax4.set_xlabel('Cum. discounted reward:\n' + str(np.round(MIDLEARNED3[k]/10,3)), fontsize=6)
    plt.xticks([])
    plt.yticks([])


    ax9 = plt.subplot(259)
    plt.imshow(img8)
    ax9.set_xlabel('Neural Network Estimation', fontsize=4)
    plt.xticks([])
    plt.yticks([])
    ax9.spines['bottom'].set_color('white')
    ax9.spines['top'].set_color('white')
    ax9.spines['right'].set_color('white')
    ax9.spines['left'].set_color('white')




    ax5 = plt.subplot(255)
    ax5.set_title("After new barrier", fontsize=8)
    plt.imshow(img5)
    ax5.set_xlabel('Cum. discounted reward:\n' + str(np.round(MIDLEARNED5[np.min((k, len2-1))]/10,3)), fontsize=6)
    plt.xticks([])
    plt.yticks([])


    ax10 = plt.subplot(2,5,10)
    plt.imshow(img10)
    ax10.set_xlabel('Neural Network Estimation', fontsize=4)
    plt.xticks([])
    plt.yticks([])
    ax10.spines['bottom'].set_color('white')
    ax10.spines['top'].set_color('white')
    ax10.spines['right'].set_color('white')
    ax10.spines['left'].set_color('white')

    plt.tight_layout(pad=0.04, w_pad=0.01, h_pad=0.01)

    #wm = plt.get_current_fig_manager()
    #wm.window.state('zoomed')

    plt.savefig('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\finalImageOutput\\' + str(k+1) + '.png', dpi=700, bbox_inches='tight')

    img = cv2.imread('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\finalImageOutput\\' + str(k+1) + '.png')

    n = img.shape

    img1 = img[0:1200, :, :]
    img2 = img[1800:2400, :, :]
    img1 = np.append(img1, img2, axis=0)
    cv2.imwrite('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\finalImageOutput\\' + str(k+1) + '.png', img1)

    print('good')

    #plt.show()
    #plt.close()

























