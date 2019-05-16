import numpy as np
import sys
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open(sys.argv[1]).convert('L')
image = np.array(image)
thr = threshold_otsu(image)
#image[np.where(image>=thr)] = 1
#image[np.where(image<thr)] = 0
Image.fromarray(255*image).show()
#print(np.min(image),np.max(image))

distance = ndi.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=image)
markers = ndi.label(local_maxi)[0]
labels = morphology.watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
