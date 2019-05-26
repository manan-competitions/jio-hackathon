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
    get(edge_im).show()
    edges = edge_im
#    get(edges).convert('RGB').save('warped.jpg')
    thr = filters.threshold_otsu(edges)
    edges[np.where(edges>=thr)] = 1
    edges[np.where(edges<thr)] = 0
    #edges = np.uint8(255*morphology.skeletonize_3d(edges))
    get(edges).show()
#    get(edges).convert('RGB').save('warped_2.jpg')
    return edge_im, edges
