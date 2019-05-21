import os
import sys
from skimage import filters, morphology, feature
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage, misc

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

config = InferenceConfig()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

get = lambda arr: Image.fromarray(arr)

def delete_masks(img, masks):
    for k in range(masks.shape[-1]):
        mask =  r["masks"][:,:,k]
        img = np.where(mask == 1, np.zeros(img.shape), img)
    return img

im = np.array(Image.open(sys.argv[1]))
get(im).show()
gray = np.array(Image.open(sys.argv[1]).convert("L"))

results = model.detect([im])
r = results[0]

masks = r["masks"]

edge_im = np.uint8(255*filters.sobel(gray))
#get(edge_im).show()
#get_s(edge_im).show()
edge_im = delete_masks(edge_im,masks)
#edge_im = filters.gaussian(edge_im, sigma=1.5)
get(edge_im).show()
image = edge_im
"""
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
    ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()
"""
edge_im = filters.gaussian(edge_im,sigma=2)
edges = np.uint8(filters.laplace(edge_im,ksize=5))
edges = filters.gaussian(edges,sigma=2)
edges = np.uint8(morphology.skeletonize_3d(edges))
lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                 line_gap=3)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Laplace edges')

ax[2].imshow(edges * 0)
print(lines[0])
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()
