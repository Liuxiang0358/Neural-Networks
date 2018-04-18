import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import cv2
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle





BLUE        = (255,0,0)
BLUE_DARK   = (150,0,0)
GREEN      = (0,255,0)
GREEN_DARK = (0,150,0)
RED       = (0,0,255)
RED_DARK  = (0,0,150)
WHITE      = (255,255,255)
BLACK      = (0,0,0)
GERY = (213, 208, 200)
GERY_DARK = (193, 186, 177)
YELLOW = (0,255,255)

FIRST_COLOR = BLUE
SECOND_COLOR = BLUE
THIRD_COLOR = BLUE
FOURTH_COLOR = BLUE




BLOCK_SIZE = 15
Frame_Size_n = 31
Frame_Size_m = 31
margin = 4
top_margin = 50

ImageMtx = np.zeros((Frame_Size_n*BLOCK_SIZE, Frame_Size_m*BLOCK_SIZE, 3))
NN_Mtx = np.ones((300, Frame_Size_m*BLOCK_SIZE, 3))*255#.astype('uint8')



print('ok')

def drawCircle(NN_Mtx, arr):
    NN_Mtx.fill(0)
    #NN_Mtx = cv2.circle(NN_Mtx, (73, 60), 30, (0, 75, 255), 3)
    #NN_Mtx = cv2.circle(NN_Mtx, (163, 60), 30, (150, 177, 210), 3)
    #NN_Mtx = cv2.circle(NN_Mtx, (253, 60), 30, (51, 64, 189), 3)
    #NN_Mtx = cv2.circle(NN_Mtx, (343, 60), 30, (38, 59, 232), 3)

    cv2.putText(NN_Mtx, 'L', (61, 70), 2, 1, WHITE)
    cv2.putText(NN_Mtx, 'R', (152, 70), 2, 1, WHITE)
    cv2.putText(NN_Mtx, 'U', (241, 70), 2, 1, WHITE)
    cv2.putText(NN_Mtx, 'D', (331, 70), 2, 1, WHITE)

    idx = np.argmax(arr)
    if idx == 0:
        NN_Mtx = cv2.circle(NN_Mtx, (73, 60+top_margin), 30, FIRST_COLOR, -3)
    if idx == 1:
        NN_Mtx = cv2.circle(NN_Mtx, (163, 60 + top_margin), 30, SECOND_COLOR, -3)
    if idx == 2:
        NN_Mtx = cv2.circle(NN_Mtx, (253, 60 + top_margin), 30, THIRD_COLOR, -3)
    if idx == 3:
        NN_Mtx = cv2.circle(NN_Mtx, (343, 60 + top_margin), 30, FOURTH_COLOR, -3)

    NN_Mtx = cv2.circle(NN_Mtx, (73, 60 + top_margin), 30, FIRST_COLOR, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (163, 60+top_margin), 30, SECOND_COLOR, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (253, 60+top_margin), 30, THIRD_COLOR, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (343, 60+top_margin), 30, FOURTH_COLOR, 3)

    cv2.putText(NN_Mtx, str(arr[0]), (54, 66+top_margin), 1, 2, YELLOW)
    cv2.putText(NN_Mtx, str(arr[1]), (144, 66+top_margin), 1, 2, YELLOW)
    cv2.putText(NN_Mtx, str(arr[2]), (234, 66+top_margin), 1, 2, YELLOW)
    cv2.putText(NN_Mtx, str(arr[3]), (324, 66+top_margin), 1, 2, YELLOW)


    NN_Mtx = cv2.circle(NN_Mtx, (50, 185+top_margin), 30, WHITE, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (130, 185+top_margin), 30, WHITE, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (210, 185+top_margin), 30, WHITE, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (290, 185+top_margin), 30, WHITE, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (370, 185+top_margin), 30, WHITE, 3)


    NN_Mtx = cv2.line(NN_Mtx, (73, 90 + margin+top_margin), (50, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (73, 90 + margin+top_margin), (130, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (73, 90 + margin+top_margin), (210, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (73, 90 + margin+top_margin), (290, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (73, 90 + margin+top_margin), (370, 155 - margin+top_margin), YELLOW, 2)

    NN_Mtx = cv2.line(NN_Mtx, (163, 90 + margin+top_margin), (50, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (163, 90 + margin+top_margin), (130, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (163, 90 + margin+top_margin), (210, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (163, 90 + margin+top_margin), (290, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (163, 90 + margin+top_margin), (370, 155 - margin+top_margin), YELLOW, 2)

    NN_Mtx = cv2.line(NN_Mtx, (253, 90 + margin+top_margin), (50, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (253, 90 + margin+top_margin), (130, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (253, 90 + margin+top_margin), (210, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (253, 90 + margin+top_margin), (290, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (253, 90 + margin+top_margin), (370, 155 - margin+top_margin), YELLOW, 2)

    NN_Mtx = cv2.line(NN_Mtx, (343, 90 + margin+top_margin), (50, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (343, 90 + margin+top_margin), (130, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (343, 90 + margin+top_margin), (210, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (343, 90 + margin+top_margin), (290, 155 - margin+top_margin), YELLOW, 2)
    NN_Mtx = cv2.line(NN_Mtx, (343, 90 + margin+top_margin), (370, 155 - margin+top_margin), YELLOW, 2)



    return NN_Mtx

#arr = np.array([3,4,2,1])
#NN_Mtx = drawCircle(NN_Mtx, arr)
#cv2.imwrite('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED4\\NN.jpg', NN_Mtx)

#print('finish')


def drawBlock(outColor, innerColar, Pos_0, Pos_1):
    for i in range(Pos_0*BLOCK_SIZE, Pos_0*BLOCK_SIZE+BLOCK_SIZE):
        for j in range(Pos_1*BLOCK_SIZE, Pos_1*BLOCK_SIZE+BLOCK_SIZE):
            ImageMtx[i, j, :] = outColor

    for i in range(Pos_0*BLOCK_SIZE+2, Pos_0*BLOCK_SIZE+BLOCK_SIZE-2):
        for j in range(Pos_1*BLOCK_SIZE+2, Pos_1*BLOCK_SIZE+BLOCK_SIZE-2):
            ImageMtx[i, j, :] = innerColar




def add_boundary(x):
    n,m = x.shape
    y = np.ones(((n+2), (m+2)))*3
    for i in range(n):
        for j in range(m):
            y[i+1][j+1] = x[i][j]
    return y

# C:\Users\lyf58\PycharmProjects\samplingTrajectory\MidLearned
# C:\Users\lyf58\PycharmProjects\samplingTrajectory\MIDLEARNED4

for k in range(127):

    ImageMtx = np.zeros((Frame_Size_n * BLOCK_SIZE, Frame_Size_m * BLOCK_SIZE, 3))

    x = np.load('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED4\\' + str(k+1) +'.npy')
    x = add_boundary(x)
    n, m = x.shape

    '''
    # only for barrier
    for i in range(19,20):
        for j in range(11,27):
            drawBlock(RED, RED_DARK, i, j)
    '''
    for i in range(n):
        for j in range(m):
            if x[i][j] == 3:
                drawBlock(RED, RED_DARK, i, j)
            elif x[i][j] == 2:
                drawBlock(GERY, GERY_DARK, i, j)
            elif x[i][j] == 1:
                drawBlock(GREEN, GREEN_DARK, i, j)

    cv2.imwrite('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED4\\' + str(k+1) + '.jpg', ImageMtx)












'''

reward_4 = np.load('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED4\\reward.npy')

for k in range(128):
    img1 = mpimg.imread('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED4\\' + str(k+1) + '.jpg')

    fig1 = plt.figure()

    ax1 = plt.subplot(151)
    ax1.set_title("First attempt", fontsize=8)
    plt.imshow(img1)
    ax1.set_xlabel('discounted reward:\n' + str(np.round(reward_4[k]/10,3)), fontsize=6)
    plt.xticks([])
    plt.yticks([])


    ax2 = plt.subplot(152)
    ax2.set_title("After 40 attempts", fontsize=8)
    plt.imshow(img1)
    # plt.axis('off')
    ax2.set_xlabel('discounted reward:\n' + str(np.round(reward_4[k]/10,3)), fontsize=6)
    plt.xticks([])
    plt.yticks([])


    ax3 = plt.subplot(153)
    ax3.set_title("After 60 attempts", fontsize=8)
    plt.imshow(img1)
    ax3.set_xlabel('discounted reward:\n' + str(np.round(reward_4[k]/10,3)), fontsize=6)
    plt.xticks([])
    plt.yticks([])

    ax4 = plt.subplot(154)
    ax4.set_title("After 80 attempts", fontsize=8)
    plt.imshow(img1)
    ax4.set_xlabel('discounted reward:\n' + str(np.round(reward_4[k]/10,3)), fontsize=6)
    plt.xticks([])
    plt.yticks([])

    ax5 = plt.subplot(155)
    ax5.set_title("After new restrict", fontsize=8)
    plt.imshow(img1)
    ax5.set_xlabel('discounted reward:\n' + str(np.round(reward_4[k]/10,3)), fontsize=6)
    plt.xticks([])
    plt.yticks([])


    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')

    plt.savefig('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\finalImageOutput\\' + str(k) + '.png', dpi=1400, bbox_inches='tight')

    #plt.show()
    #plt.close()


'''
