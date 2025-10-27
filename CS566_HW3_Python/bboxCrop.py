from skimage import color
import numpy as np


def bbox_crop(img):
    """
    Remove border rows and columns which are all zeros.
    """
    if img.ndim == 2 or img.shape[2] == 1:
        gray_img = img if img.ndim == 2 else img[:, :, 0]
    else:
        gray_img = color.rgb2gray(img)

    rows_nz, cols_nz = np.nonzero(gray_img != 0)

    cropped = img[min(rows_nz):max(rows_nz)+1,
                  min(cols_nz):max(cols_nz)+1, ...]
    return cropped
