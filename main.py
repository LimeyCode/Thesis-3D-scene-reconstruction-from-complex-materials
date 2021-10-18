# This is a sample Python script.
import configparser
import numpy
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from open3d import *

from tempfile import TemporaryFile

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))

def pointcloud(depth, fov):
    fy = 0.5 / np.tan(fov * 0.5)  # assume aspectRatio is one.
    fx = fy/1.219419924
    height = depth.shape[0]
    width = depth.shape[1]

    mask = np.where(depth > 0)

    x = mask[1]
    y = mask[0]

    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = (y.astype(np.float32) - height * 0.5) / height

    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = -depth[y, x]
    ones = np.ones(world_z.shape[0], dtype=np.float32)

    #return  np.array(world_x,world_y,world_z)
    #return [world_x,world_y,world_z]
    return np.vstack((world_x, world_y, world_z)).T
    #return np.vstack((world_x, world_y, world_z, ones)).T

def convert_from_uvd(self, u, v, d):
    d *= self.pxToMetre
    x_over_z = (self.cx - u) / self.focalx
    y_over_z = (self.cy - v) / self.focaly
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    return x, y, z

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# Load txt file of cords for camera and light positions
cameraPositionsTxt = os.path.join("cords.txt")
arrayCameraPositions = np.loadtxt(cameraPositionsTxt)
print(arrayCameraPositions)

# Create hdf5 file to store camera positions
with h5py.File('CameraPositions.hdf5', 'w') as f:
    dset = f.create_dataset("Left_Camera_positions", data = cameraPositionsTxt)
    #dset = f.create_dataset("Right_Camera_positions", data= cameraPositionsTxt)

# Open file to read
CameraPositionsFile = h5py.File('CameraPositions.hdf5', 'r')

# List all keys of the hdf5 file
listOfKeys = list(CameraPositionsFile.keys())
CameraPositionsFile.visititems(print_attrs)
print(listOfKeys)

cameraPositionArray = CameraPositionsFile.get('Camera_positions')
CameraPositions = np.array(cameraPositionArray)
CameraPositionsFile.close()

# Look at EPI images
EPIdisparityFile = h5py.File('disparity_map.hdf5', 'r')
#EPItwoDisparityFile = h5py.File('disparity_mapTwo.hdf5', 'r')
# Print Attributes of file
EPIdisparityFile.visititems(print_attrs)
EPIdisparityFileKeys = list(EPIdisparityFile.keys())
#EPItwoDisparityFileKeys = list(EPIdisparityFile.keys())
# Get the whole volume of disparity
EPIvolume = np.array(EPIdisparityFile.get('disparity_map'))
#EPIvolumeTwo = np.array(EPItwoDisparityFile.get('disparity_map'))
# Print volume shape
print("shape of array : ", EPIvolume.shape)

# Get the first image of the volume
firstImage = (EPIvolume[49, :, :])
firstPixel = firstImage[0][0]

# Convert disparity to depth
DisparityToDepth = ((3.048 * 1101.389) / firstImage)
# Find the xyz of points from the EPI disparity map
testOne = pointcloud(DisparityToDepth, 39.6)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(testOne)
o3d.io.write_point_cloud("./data.ply", pcd)
o3d.visualization.draw_geometries([pcd])


# Get Helmholtz images

# Get up camera images
upHSimagesFile = h5py.File('upHS.hdf5', 'r')
upHSimagesFile.visititems(print_attrs)
upImagePixels = np.array(upHSimagesFile.get('lightfield'))

# Get down camera images
downHSimagesFile = h5py.File('DownHS.hdf5', 'r')
downHSimagesFile.visititems(print_attrs)
downImagePixels = np.array(downHSimagesFile.get('lightfield'))

# Loop through images and get corresponding pixel values
"""
Variables needed
Ol : position of camera left image
Or : position of camera right image
Vl : vector left image
Vr : vector right image
P  : position of point
Ir : pixel intensity right image
Il : pixel intensity left image

W  : matrix of w vectors
n  : normal vectors
"""
EPIvolume = EPIvolume[:, 250:966, :]
#EPIvolumeTwo = EPIvolumeTwo[:, 0:716, :]
upImagePixels = upImagePixels[:, 250:966, :]
downImagePixels = downImagePixels[:, 0:716, :]
i = 0
j = 0
k = 0
loopCounter = 0
matrixW = numpy.zeros(shape= (716, 101, 793, 3))
while i < 100:
    EPIimage = EPIvolume[i, :, :]
   # EPIimageTwo = EPIvolumeTwo[i, :, :]

    UpTmpImage = upImagePixels[i, :, :]
    DownTmpImage = downImagePixels[i, :, :]

    DisparityToDepth = ((3.048 * 50) / EPIimage)
   # DisparityToDepth = ((3.048 * 50) / EPIimageTwo)

    EPIcords = pointcloud(EPIimage, 36)
  #  EPIcordsTwo = pointcloud(EPIimageTwo, 36)

    Ol = arrayCameraPositions[i]
    Or = arrayCameraPositions[101 + i]

    print(i)
    i += 1
    loopCounter = 0
    j = 0
    while j < 716:
        j += 1
        k = 0
        while k < 793:
            # Get pixel intensities Ol and Or
            il = UpTmpImage[j-1][k]
            ir = DownTmpImage[j-1][k]
            # Convert EPI disparity image into cords
            P = EPIcords[loopCounter]
            #Ptwo = EPIcordsTwo[loopCounter]

            OlMinusP = Ol - P
            OrMinusP = Or - P

            NormOlMinusP = normalize(OlMinusP)
            NormOrMinusP = normalize(OrMinusP)

            Vl = OlMinusP/NormOlMinusP
            Vr = OrMinusP/NormOrMinusP
            #print(loopCounter)
            loopCounter += 1
            OrNormSquared = NormOrMinusP**2
            OlNormSquared = NormOlMinusP**2
            leftSidew = il*(Vl/OlNormSquared)
            rightSidew = ir*(Vr/(OrNormSquared))
            w = leftSidew - rightSidew
            matrixW[j-1][i-1][k] = w
            k += 1


outfile = os.path.join("W_Matrix")
np.save(outfile, matrixW)
print('end')


# Find the xyz of a point at a certain pixel

# get camera position

# Find this point for light up setup
# Find this point for light down setup

# Look at light up setup
# Find position of camera Ol
# Find intensity of the point
# Calculate Vl

# Look at light down setup
# Find position of camera Or
# Find intensity of the point
# Calculate Vr

# Calculate vector
# Add to matrix W
# Repeat for all pixels on object
# Go to next image repeat
# calculate norm for each pair
# Average all normals
# reconstruct object








# See PyCharm help at https://www.jetbrains.com/help/pycharm/
