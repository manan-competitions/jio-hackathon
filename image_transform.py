import sys
from skimage import filters, morphology, feature
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

scale = lambda arr,m,M: m + (M-m)*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
# lambda function get a PIL image from an array
get = lambda arr: Image.fromarray(scale(arr,0,255))

im = np.array(Image.open(sys.argv[1]).convert("L"))
#get(im).show()
edge_im = filters.gaussian(im,sigma=1)
get(edge_im).show()
edges = np.uint8(255*filters.sobel(edge_im))
edges = filters.gaussian(edges,sigma=1)
get(edges).show()
edges = edge_im
get(edges).convert('RGB').save('warped.jpg')
thr = filters.threshold_otsu(edges)
edges[np.where(edges>=thr)] = 1
edges[np.where(edges<thr)] = 0
#edges = np.uint8(255*morphology.skeletonize_3d(edges))
get(edges).show()
get(edges).convert('RGB').save('warped_2.jpg')
lines = probabilistic_hough_line(edges, threshold=25, line_length=20,
                                 line_gap=1)
image = edges
"""
fig, axes = plt.subplots(1, 1, figsize=(15, 5), sharex=True, sharey=True)
canvas = FigureCanvas(fig)

for line in lines:
    p0, p1 = line
    try:
        slope = (p1[1]-p0[1])/(p1[0]-p0[0])
        angle = np.arctan(slope)*180/np.pi
    except:
        slope = 999
        angle = (np.pi/4)*180/np.pi
    if 45 <= abs(angle) <= 135:
        axes.plot((p0[0], p1[0]), (p0[1], p1[1]),'k-')
axes.set_xlim((0, image.shape[1]))
axes.set_ylim((image.shape[0], 0))
axes.set_title('Probabilistic Hough')
plt.show()
"""
"""
get_line_param = lambda l: ((l[1][1]-l[0][1])/(l[1][0]-l[0][0]),l[0][1]+l[0][0]*(l[1][1]-l[0][1])/(l[1][0]-l[0][0]))
"""
"""
def intersect(p1,p2,p3,p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    return (((x2*y1-x1*y2)*(x4-x3)-(x4*y3-x3*y4)*(x2-x1))/((x2-x1)*(y4-y3)-(x4-x3)*(y2-y1)),
           ((x2*y1-x1*y2)*(y4-y3)-(x4*y3-x3*y4)*(y2-y1))/((x2-x1)*(y4-y3)-(x4-x3)*(y2-y1)))

tol = 142
intersection_pts = []
print(len(lines))
for line1 in lines[:]:
    for line2 in lines[:]:
        try:
            pt = intersect(line1[0],line1[1],line2[0],line2[1])
        except:
            continue
        if intersection_pts == []:
            intersection_pts.append([pt,1])
        else:
            for i_pt in intersection_pts:
                if dist(pt,i_pt) <= tol and i_pt[1] <= 0:
                    intersection_pts[intersection_pts.index(i_pt)][1] += 1
                    break
            else:
                intersection_pts.append([pt,1])

from pprint import pprint
intersection_pts.sort(key = lambda x: x[1])
print(intersection_pts[-1])
pt = intersection_pts[-1][0]
l_pt = (0,450)
r_pt = (600,450)
"""
"""
dist = lambda p1,p2: (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
get_min_dist = lambda l1,l2: np.min([dist(l1[0],l2[0]), dist(l1[0],l2[1]), dist(l1[1],l2[0]), dist(l1[1],l2[1])])
get_max_dist_pts = lambda l1,l2: np.argmax([dist(l1[0],l2[0]), dist(l1[0],l2[1]), dist(l1[1],l2[0]), dist(l1[1],l2[1])])
tol = 100
new_lines = []
best_lines = []

for line1 in lines:
    for line2 in lines:
        if get_min_dist(line1,line2) <= tol:
            if not pts:
                ind = get_max_dist_pts(line1,line2)
            elif get_min_dist:
                pass
"""
