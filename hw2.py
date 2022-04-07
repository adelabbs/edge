import sys
import os
import cv2 as cv
import numpy as np
import math

def checkIfFileExists(filename):
    return os.path.isfile(filename)


def readImage(filename):
    return cv.imread(filename, cv.IMREAD_GRAYSCALE)


def writeImage(img, filename):
    cv.imwrite(filename, img)

def processImage(filename, data="./", out="./"):
    print("Processing ", filename)
    if filename.endswith("jpg"):
        img0 = readImage(os.path.join(data, filename))
        #print("Padding image for testing...")
        #img1 = padImage(img0, 50, 50)
        
        h = getGaussianKernel(1)

        img1 = myImageFilter(img0, h)
        writeImage(img1, os.path.join(out, filename+"_my.jpg"))

        img1 = cv.GaussianBlur(img0, (7, 7), 1)
        writeImage(img1, os.path.join(out, filename+"_cv.jpg"))


def zeroPadImage(img, rowPadding, colPadding):
    shape = np.shape(img)
    rows = shape[0] + 2 * rowPadding 
    cols = shape[1] + 2 * colPadding
    paddedImg = np.zeros((rows, cols))
    paddedImg[rowPadding: rowPadding + shape[0], colPadding: colPadding + shape[1]] = img
    return paddedImg

def padImage(img, rowPadding, colPadding):
    shape = np.shape(img)
    rows = shape[0] + 2 * rowPadding # pad on top and bottom
    cols = shape[1] + 2 * colPadding # pad on the left and on the right

    paddedImg = np.zeros((rows, cols))

    # Compute padding values
    top = np.tile(img[0, :], (rowPadding, 1))
    bottom = np.tile(img[-1, :], (rowPadding, 1))
    left = np.tile(img[:, 0], (colPadding, 1)).T
    right = np.tile(img[:, -1], (colPadding, 1)).T
    top_left = np.ones((rowPadding, colPadding)) * img[0][0]
    top_right = np.ones((rowPadding, colPadding)) * img[0][-1]
    bottom_left = np.ones((rowPadding, colPadding)) * img[-1][0]
    bottom_right = np.ones((rowPadding, colPadding)) * img[-1][-1]

    # top
    paddedImg[0:rowPadding, 0:colPadding] = top_left
    paddedImg[0:rowPadding, colPadding :colPadding + shape[1]] = top
    paddedImg[0:rowPadding, colPadding + shape[1]:] = top_right

    # center
    paddedImg[rowPadding:rowPadding + shape[0], :colPadding] = left
    paddedImg[rowPadding: rowPadding + shape[0], colPadding: colPadding + shape[1]] = img
    paddedImg[rowPadding:rowPadding + shape[0], colPadding + shape[1]:] = right

    # bottom
    paddedImg[rowPadding+shape[0]:, :colPadding] = bottom_left
    paddedImg[rowPadding+shape[0]:,colPadding:colPadding + shape[1]] = bottom
    paddedImg[rowPadding+shape[0]:, colPadding+ shape[1]:] = bottom_right

    return paddedImg


"""
Performs convolution of the image with the provided kernel h
The image should be a grayscale image (i.e. a 2D numpy array)
and the provided kernel matrix h is assumed to have odd dimensions
in both directions
"""
def myImageFilter(img, h):
    # Padding values depend on the kernel dimensions
    hshape = np.shape(h)
    ishape = np.shape(img)
    rowPadding = hshape[0] // 2
    colPadding = hshape[1] // 2
    padded = padImage(img, rowPadding, colPadding)

    output = np.zeros(ishape)

    for y in range(rowPadding, ishape[0] + rowPadding):
        for x in range(colPadding, ishape[1] + colPadding):
            roi = padded[y - rowPadding:y + rowPadding + 1, x - colPadding:x + colPadding + 1]
            s = (roi * h).sum()
            output[y - rowPadding, x - colPadding] = s
    return output

def getGaussianKernel(sigma):
    hsize = 2 * math.ceil(3 * sigma) + 1
    x = np.linspace(- (hsize//2), hsize // 2, hsize)
    xy, yx = np.meshgrid(x, x) # returns 2 matrices of distances to the
                                # center pixel on the x and y axis
    ii = np.square(xy)
    jj = np.square(yx)
    kernel = np.exp(-0.5 * (ii + jj) / (sigma**2))
    kernel = kernel / (2 * math.pi * sigma**2)
    return kernel 

def main():
    DIR = "./dataset"
    OUT = "./out"  # intermediate output images directory
    directory = os.fsencode(DIR)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        processImage(filename, data=DIR, out=OUT)

if __name__ == "__main__":
    main()
