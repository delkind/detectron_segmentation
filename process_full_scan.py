import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np


def split_image(image_path, crop_size, border_size):
    image = cv2.imread(image_path)
    return create_crops_list(border_size, crop_size, image), image


def create_crops_list(border_size, crop_size, image):
    crop_coords = create_crops_coords_list(crop_size, border_size, image)
    crops = [image[i:i + crop_size, j:j + crop_size, ...] for (i, j) in crop_coords]
    return list(zip(crops, crop_coords))


def create_crops_coords_list(crop_size, border_size, image):
    vert = list(range(0, image.shape[0], crop_size - 2 * border_size))
    horiz = list(range(0, image.shape[1], crop_size - 2 * border_size))
    vert = list(filter(lambda v: v + crop_size <= image.shape[0], vert)) + [image.shape[0] - crop_size]
    horiz = list(filter(lambda v: v + crop_size <= image.shape[1], horiz)) + [image.shape[1] - crop_size]
    crop_coords = list(itertools.product(vert, horiz))
    return crop_coords


def filter_rois(rois, border_size, crop_size):
    valid = list(map(lambda r: not (at_border((r[0], r[2]), border_size, crop_size) or at_border((r[1], r[3]),
                                                                                                 border_size,
                                                                                                 crop_size)), rois))
    # return np.nonzero(valid)[0]
    return range(len(rois))


def at_border(r, border_size, crop_size):
    return (r[0] < border_size and r[1] < border_size) or (
            r[0] > crop_size - border_size and r[1] > crop_size - border_size)


if __name__ == '__main__':
    crops, image = split_image('data/full-scans/297894255_79_cropped.png', 320, 20)
    for crop, coords in crops:
        processed = np.concatenate([crop, cv2.equalizeHist(crop)], axis=1)
        plt.imshow(processed, cmap='gray')
        plt.show()

    pass
