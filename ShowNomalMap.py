import configparser
import numpy
import numpy as np
import h5py
import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from tempfile import TemporaryFile
from numpy.linalg import matrix_rank
from itertools import product
import pandas as pd

normalMatrix = np.load("NormalMatrix.npy")

uMatrix = numpy.zeros(shape=(716, 793))
vMatrix = numpy.zeros(shape=(716, 793))
zMatrix = numpy.zeros(shape=(716, 793))
o = 0
p = 0


while o < 716:
    o += 1
    p = 0
    print(o)
    while p < 793:
        uMatrix[o - 1][p] = normalMatrix[o - 1][p][0]
        vMatrix[o - 1][p] = normalMatrix[o - 1][p][1]
        zMatrix[o - 1][p] = normalMatrix[o - 1][p][2]
        normalMatrix[o - 1][p][0] = ((((normalMatrix[o - 1][p][0])) + 1) / 2)
        normalMatrix[o - 1][p][1] = ((((normalMatrix[o - 1][p][1])) + 1) / 2)
        normalMatrix[o - 1][p][2] = ((((normalMatrix[o - 1][p][2])) + 1) / 2)
        p += 1



print(normalMatrix.shape)
matplotlib.pyplot.imshow(normalMatrix)
plt.imshow(normalMatrix)
plt.show()
plt.imsave('NormalImage.png', normalMatrix)

