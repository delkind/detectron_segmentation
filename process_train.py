from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np

import cv2

pics = [join('data/train', f) for f in listdir('data/train') if isfile(join('data/train', f)) and f.lower().endswith('.jpg')]

for pic in pics:
    image = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    if image.shape[1] != image.shape[0]:
        image = image[:,:image.shape[0], ...]
        image = np.concatenate([image, cv2.equalizeHist(image)], axis=1)
        plt.imshow(image, cmap='gray')
        plt.show()
