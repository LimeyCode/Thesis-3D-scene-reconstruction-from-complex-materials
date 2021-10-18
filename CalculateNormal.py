import numpy
import numpy as np
import os


# Load the W matrix
wMatrix = np.load("W_Matrix.npy")

i = 0
j = 0
k = 0

B = wMatrix[50,:,0]
u, s, vh = np.linalg.svd(B, full_matrices=True)

normalMatrix = numpy.zeros(shape=(716, 793, 3))
while i < 716:
    i += 1
    j = 0
    #print(i)
    while j < 793:
        tmpNorm = wMatrix[i-1, :, j]
        u, s, vh = np.linalg.svd(tmpNorm, full_matrices=True)
        vh_transpose = vh.transpose()
        normalMatrix[i-1][j] = vh_transpose[2]
        j += 1

outfile = os.path.join("NormalMatrix")
np.save(outfile, normalMatrix)

print("end")
