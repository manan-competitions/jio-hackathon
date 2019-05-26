import os
import sys
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

from rectification import rectify_image
from image_transform import pre_process

ROOT_DIR = os.path.abspath("./")
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
#io.imshow(final_im)
#plt.show()
image = final_im

# Classic straight-line Hough transform
h, theta, d = hough_line(image)

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=cm.gray, aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')
ax[2].imshow(image, cmap=cm.gray)

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
    slope = abs(np.arctan((y1)/y0)*180/np.pi)
    if 75 <= slope <= 90:
        print(slope)
        ax[2].plot((0, image.shape[1]), (y0, y1), '-r')

ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()
