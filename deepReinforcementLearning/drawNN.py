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
left_margin = 20




# NN_Mtx = np.ones((300, Frame_Size_m*BLOCK_SIZE, 3))*255#.astype('uint8')
NN_Mtx = np.zeros((300, Frame_Size_m*BLOCK_SIZE, 3))


print('ok')

def drawCircle(NN_Mtx, arr):
    NN_Mtx.fill(255)
    #NN_Mtx = cv2.circle(NN_Mtx, (73, 60), 30, (0, 75, 255), 3)
    #NN_Mtx = cv2.circle(NN_Mtx, (163, 60), 30, (150, 177, 210), 3)
    #NN_Mtx = cv2.circle(NN_Mtx, (253, 60), 30, (51, 64, 189), 3)
    #NN_Mtx = cv2.circle(NN_Mtx, (343, 60), 30, (38, 59, 232), 3)

    cv2.putText(NN_Mtx, 'L', (61+left_margin, 70), 2, 1, BLACK)
    cv2.putText(NN_Mtx, 'R', (152+left_margin, 70), 2, 1, BLACK)
    cv2.putText(NN_Mtx, 'U', (241+left_margin, 70), 2, 1, BLACK)
    cv2.putText(NN_Mtx, 'D', (331+left_margin, 70), 2, 1, BLACK)

    idx = np.argmax(arr)
    if idx == 0:
        NN_Mtx = cv2.circle(NN_Mtx, (73+left_margin, 60+top_margin), 30, FIRST_COLOR, -3)
    if idx == 1:
        NN_Mtx = cv2.circle(NN_Mtx, (163+left_margin, 60 + top_margin), 30, SECOND_COLOR, -3)
    if idx == 2:
        NN_Mtx = cv2.circle(NN_Mtx, (253+left_margin, 60 + top_margin), 30, THIRD_COLOR, -3)
    if idx == 3:
        NN_Mtx = cv2.circle(NN_Mtx, (343+left_margin, 60 + top_margin), 30, FOURTH_COLOR, -3)

    NN_Mtx = cv2.circle(NN_Mtx, (73+left_margin, 60 + top_margin), 30, FIRST_COLOR, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (163+left_margin, 60+top_margin), 30, SECOND_COLOR, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (253+left_margin, 60+top_margin), 30, THIRD_COLOR, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (343+left_margin, 60+top_margin), 30, FOURTH_COLOR, 3)


    cv2.putText(NN_Mtx, str(np.round(arr[0]/ 10, 0).astype(int)), (54+left_margin, 66+top_margin), 1, 1.5, BLACK)
    cv2.putText(NN_Mtx, str(np.round(arr[1]/ 10, 0).astype(int)), (144+left_margin, 66+top_margin), 1, 1.5, BLACK)
    cv2.putText(NN_Mtx, str(np.round(arr[2]/ 10, 0).astype(int)), (234+left_margin, 66+top_margin), 1, 1.5, BLACK)
    cv2.putText(NN_Mtx, str(np.round(arr[3]/ 10, 0).astype(int)), (324+left_margin, 66+top_margin), 1, 1.5, BLACK)


    NN_Mtx = cv2.circle(NN_Mtx, (50+left_margin, 185+top_margin), 30, BLACK, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (130+left_margin, 185+top_margin), 30, BLACK, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (210+left_margin, 185+top_margin), 30, BLACK, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (290+left_margin, 185+top_margin), 30, BLACK, 3)
    NN_Mtx = cv2.circle(NN_Mtx, (370+left_margin, 185+top_margin), 30, BLACK, 3)


    NN_Mtx = cv2.line(NN_Mtx, (73+left_margin, 90 + margin+top_margin), (50+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (73+left_margin, 90 + margin+top_margin), (130+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (73+left_margin, 90 + margin+top_margin), (210+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (73+left_margin, 90 + margin+top_margin), (290+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (73+left_margin, 90 + margin+top_margin), (370+left_margin, 155 - margin+top_margin), BLACK, 2)

    NN_Mtx = cv2.line(NN_Mtx, (163+left_margin, 90 + margin+top_margin), (50+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (163+left_margin, 90 + margin+top_margin), (130+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (163+left_margin, 90 + margin+top_margin), (210+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (163+left_margin, 90 + margin+top_margin), (290+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (163+left_margin, 90 + margin+top_margin), (370+left_margin, 155 - margin+top_margin), BLACK, 2)

    NN_Mtx = cv2.line(NN_Mtx, (253+left_margin, 90 + margin+top_margin), (50+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (253+left_margin, 90 + margin+top_margin), (130+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (253+left_margin, 90 + margin+top_margin), (210+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (253+left_margin, 90 + margin+top_margin), (290+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (253+left_margin, 90 + margin+top_margin), (370+left_margin, 155 - margin+top_margin), BLACK, 2)

    NN_Mtx = cv2.line(NN_Mtx, (343+left_margin, 90 + margin+top_margin), (50+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (343+left_margin, 90 + margin+top_margin), (130+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (343+left_margin, 90 + margin+top_margin), (210+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (343+left_margin, 90 + margin+top_margin), (290+left_margin, 155 - margin+top_margin), BLACK, 2)
    NN_Mtx = cv2.line(NN_Mtx, (343+left_margin, 90 + margin+top_margin), (370+left_margin, 155 - margin+top_margin), BLACK, 2)



    return NN_Mtx







x = np.load('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED4\\output.npy')

for i in range(113):

    vector = x[i*4 : (i+1)*4]

    print(vector)

    drawCircle(NN_Mtx, vector)
    # drawCircle(NN_Mtx, np.round(vector / 10, 0).astype(int))

    cv2.imwrite('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED4\\NN' + str(i+1) +'.jpg', NN_Mtx)

    print('ok')




drawCircle(NN_Mtx, np.array([0,0,0,0]))
cv2.imwrite('C:\\Users\\lyf58\\PycharmProjects\\samplingTrajectory\\MIDLEARNED4\\NN114.jpg', NN_Mtx)



