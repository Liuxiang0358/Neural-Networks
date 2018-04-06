import numpy as np
from PIL import Image
import scipy.misc




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



Width = 256
Height = 256
mupathLen = 35
SamplingRatio = 0.2

for i in range(200000):
    T = muPathMaskGen(mupathLen, Width, Height, SamplingRatio)
    scipy.misc.imsave("D:\\myPythonProjects\\ImageDenoising\\masks\\" + str(i) + ".png", T)
    #T = np.asarray(T, dtype=bool)
    #np.savetxt("D:\\myPythonProjects\\ImageDenoising\\masks\\" + str(i) + ".csv", T)
























