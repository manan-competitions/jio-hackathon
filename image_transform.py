import sys
from skimage import filters, morphology, feature
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)

from PIL import Image
import numpy as np

scale = lambda arr,m,M: m + (M-m)*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
# lambda function get a PIL image from an array
get = lambda arr: Image.fromarray(scale(arr,0,255))

def pre_process(im):
    im = np.array(Image.fromarray(np.uint8(scale(im,0,255))).convert("L"))
    edge_im = filters.gaussian(im,sigma=1)
#    get(edge_im).show()
    #edge_im = np.uint8(255*filters.sobel(edge_im))
    edge_im = filters.gaussian(edge_im,sigma=1)
    # get(edge_im).show()
    edges = np.copy(edge_im)
#    get(edges).convert('RGB').save('warped.jpg')
    thr = filters.threshold_otsu(edges)
    edges[np.where(edges>=thr)] = 1
    edges[np.where(edges<thr)] = 0
    #edges = np.uint8(255*morphology.skeletonize_3d(edges))
    # get(edges).show()
#    get(edges).convert('RGB').save('warped_2.jpg')
    return edge_im, edges


def hist_segment(im):
    n = 10
    l = 256
    hist, bin_edges = np.histogram(im, bins=256)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    plt.plot(hist)
    binary_img = im > 0.5
    return binary_img

def visualize_hsv(im):
    pixel_colors = im.reshape((np.shape(im)[0]*np.shape(im)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_im)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

def show_hsv(im):
    import matplotlib.pyplot as plt

    from skimage import data
    from skimage.color import rgb2hsv
    rgb_img = im
    hsv_img = rgb2hsv(rgb_img)
    hue_img = hsv_img[:, :, 0]
    sat_img = hsv_img[:,:, 1]
    value_img = hsv_img[:, :, 2]

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(8, 2))

    ax0.imshow(rgb_img)
    ax0.set_title("RGB image")
    ax0.axis('off')
    ax1.imshow(hue_img, cmap='hsv')
    ax1.set_title("Hue channel")
    ax1.axis('off')
    ax2.imshow(value_img)
    ax2.set_title("Value channel")
    ax2.axis('off')
    ax3.imshow(sat_img)
    ax3.set_title("Saturation channel")
    ax3.axis('off')

    fig.tight_layout()
    plt.show()

