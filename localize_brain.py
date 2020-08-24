import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import Polygon

from rect import Rect

__brain_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def find_largest_polygon(mask):
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ctrs = sorted(ctrs, key=lambda c: cv2.contourArea(c, oriented=False), reverse=True)
    return ctrs[0].squeeze()


def detect_brain(image):
    ret, thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    ctrs = find_largest_polygon(thresh)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [ctrs], color=255)
    mask = cv2.erode(mask, __brain_dilation_kernel, cv2.BORDER_CONSTANT)
    mask = cv2.dilate(mask, __brain_dilation_kernel, cv2.BORDER_CONSTANT)
    ctrs = find_largest_polygon(mask)
    return mask, Rect(*cv2.boundingRect(ctrs)), ctrs


def main():
    image = cv2.imread('output/experiments/cache/129564675/65-thumbnail.jpg', cv2.IMREAD_GRAYSCALE)
    brain, bbox, poly = detect_brain(image)
    # cv2.rectangle(brain, *bbox.corners(), color=255)
    plt.imshow(brain, cmap='gray')
    plt.show()
    print(Polygon(poly.tolist()).area)


if __name__ == '__main__':
    main()
