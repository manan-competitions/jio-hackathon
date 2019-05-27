#!/bin/python
import os
import sys
ROOT_DIR = os.path.abspath("./")
import random
import math
import numpy as np
from scipy import ndimage, misc
import skimage.io as io
from skimage.transform import (hough_line, hough_line_peaks,
                                probabilistic_hough_line)
from skimage import filters, morphology, feature
import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
sys.path.append(os.path.join(ROOT_DIR, "Image-rectification/"))
<<<<<<< HEAD
from ImageRectification.rectification import rectify_image
from image_transform import pre_process, show_hsv, visualize_hsv
=======
from rectification import rectify_image
from image_transform import pre_process
>>>>>>> 49243ba799410da687d3476ec16c01f629cd056a

sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, "coco/"))
import coco
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    DETECTION_MAX_INSTANCES = 200
    DETECTION_MIN_CONFIDENCE = 0.35
    IMAGES_PER_GPU = 1

# lambda function to scale an array to [min,max]
scale = lambda arr,min,max: min + (max-min)*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
# lambda function get a PIL image from an array
get = lambda arr: Image.fromarray(np.uint8(scale(arr,0,255)))

print('Loading mask RCNN model...')
config = InferenceConfig()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

def get_dominant_color(img):
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    return palette[np.argmax(counts)]

def delete_masks(img, masks):
    img = np.copy(img)
    dom_col = get_dominant_color(img)
    for k in range(masks.shape[-1]):
        mask =  r["masks"][:,:,k]
        for c in range(3):
            img[:,:,c] = np.where(mask == 1, dom_col[c]*np.ones(img[:,:,c].shape), img[:,:,c])
    return img

#def get_rectangles()

## -- MAIN -- ##

print('Reading Image...')
im = io.imread(sys.argv[1]).astype("int32")
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
print('Detecting objects and masking them...')
results = model.detect([im], verbose=1)
r = results[0]

masks = r["masks"]

im_without_cars = delete_masks(im,masks)
print('Pre-processing masked image...')
print(im_without_cars.shape)
proc_im, thr_proc_im = pre_process(im_without_cars)

print(proc_im.shape,thr_proc_im.shape)
print('Rectifying perspective of image')

final_im = rectify_image(proc_im, 4, algorithm='independent')
ax1.imshow(im)
ax1.set_title("Initial Image")
ax1.axis('off')
ax2.imshow(im_without_cars)
ax2.set_title("Masked Images")
ax2.axis('off')
ax3.imshow(thr_proc_im)
ax3.set_title("Filtered and Proccessed Image")
ax3.axis('off')
ax4.imshow(final_im)
ax4.set_title("Birds Eye View")
ax4.axis('off')

plt.show()
plt.imshow(final_im)
plt.show()
# Uncomment for additional visualizations
#
#
# show_hsv(im)
# visualize_hsv(im)
# =======
# final_im = rectify_image(thr_proc_im, 4, algorithm='independent')
# #ax1.imshow(im)
# #ax2.imshow(im_without_cars)
# #ax3.imshow(final_im,cmap='gray')
# #ax4.imshow(thr_proc_im,cmap='gray')

# #ax1.imsave(im,)
# #get(im_without_cars).show()
# get(final_im).show()
# #get(thr_proc_im).show()
# #plt.show()
# >>>>>>> 49243ba799410da687d3476ec16c01f629cd056a
