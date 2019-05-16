import numpy as np
import sys
from skimage import morphology,measure,filters
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
import scipy
from PIL import Image
import matplotlib.pyplot as plt
import cv2

image = Image.open(sys.argv[1]).resize((1080,768))
image = np.array(image)
"""
r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]
r = filters.median(r)
g = filters.median(g)
b = filters.median(b)
#r = filters.gaussian(r,sigma=0.01)
#g = filters.gaussian(g,sigma=0.01)
#b = filters.gaussian(b,sigma=0.01)
image[:,:,0] = r
image[:,:,1] = g
image[:,:,2] = b
print(image.shape)
"""
Image.fromarray(image).show()
#image = image.reshape(image.shape[0]*image.shape[1],3)
num_bins = 16
img = np.array(image/(256/num_bins)).astype(int)
bins = np.zeros((num_bins,num_bins,num_bins))
#np.add.at(bins,img,1)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        ind = np.uint8(image[i,j,:]/(256/num_bins))
        bins[ind[0],ind[1],ind[2]] += 1

print(bins,bins.shape)
max_bin = np.where(bins == np.max(bins))
print(max_bin)

new_im = np.zeros((image.shape[0],image.shape[1],3))
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        ind = np.uint8(image[i,j,:]/(256/num_bins))
        if abs(ind-max_bin).all() == 0:
            new_im[i,j,:] = image[i,j,:]
        else:
            new_im[i,j,:] = [0,0,0]

new_im  = np.uint8(new_im)
#Image.fromarray(new_im).show()

hull_initial = np.array(Image.fromarray(new_im).convert('L'))
Image.fromarray(new_im).show()
#gray = cv2.cvtColor(hull_initial, cv2.COLOR_BGR2GRAY) # convert to grayscale
blur = cv2.blur(hull_initial, (3, 3)) # blur the image
ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# create hull array for convex hull points
hull = []
# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))

# create an empty black image
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(drawing, hull, i, color, 1, 8)

cv2.imshow('image',drawing)
cv2.waitKey(0)
