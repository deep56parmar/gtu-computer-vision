'''Histograms'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../practical-1/demo.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()
img = cv2.imread('../practical-1/demo.jpg', 0)
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))
cv2.imwrite('res.png', res)
