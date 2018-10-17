import matplotlib.pyplot as plt

import cv2
from skimage.feature import hog
from skimage import data, color, exposure
import numpy

# file1='000200.jpg'
file1='IMG.jpg'

# print(data.astronaut())
x = cv2.imread(file1).shape
# image = color.rgb2gray(x)
image = color.rgb2gray(data.astronaut())

im = cv2.imread(file1)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = cv2.resize(im,(512,512))
fd, hog_image = hog(im, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

print(type(fd),hog_image.shape)
dist = numpy.linalg.norm(fd-fd)
print(dist)
# exit(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
ax1.imshow(im, cmap=plt.cm.gray)

ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
cv2.imwrite('tmp.jpg',hog_image_rescaled)

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()