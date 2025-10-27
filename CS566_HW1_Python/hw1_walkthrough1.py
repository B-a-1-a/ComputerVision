# -------------------------------------------------------------------------
# Part 1 - Create a Vincent van Gogh collage
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def hw1_walkthrough1():
    # Load the image "Vincent_van_Gogh.png" into memory
    img = io.imread("Vincent_van_Gogh.png")

    # Note the image type, shape, and range of values
    print(type(img))
    print(img.shape)
    print(img.dtype)
    print(img.max())

    # Display the image
    # plt.figure()
    # plt.imshow(img)
    # plt.title("Original Image")

    # Separate the image into three color channels and store each channel into
    # a new image

    red_channel = img[:, :, 0]
    red_image = np.zeros_like(img)
    red_image[:, :, 0] = red_channel
    # plt.figure()
    # plt.imshow(red_image)
    # plt.title("Red image")

    #
    # Similarly extract green_channel and blue_channel and create green_image
    # and blue_image
    green_channel = img[:, :, 1]
    green_image = np.zeros_like(img)
    green_image[:, :, 1] = green_channel
    # plt.figure()
    # plt.imshow(green_image)
    # plt.title("Green image")

    blue_channel = img[:, :, 2]
    blue_image = np.zeros_like(img)
    blue_image[:, :, 2] = blue_channel
    # plt.figure()
    # plt.imshow(blue_image)
    # plt.title("Blue image")

    # Create a 1 x 4 image collage in the following arrangement
    #
    # original image | red channel | green channel  | blue channel

    collage_1x4 = np.concatenate(
        (img, red_image, green_image, blue_image), axis=1)
    # plt.figure()
    # plt.imshow(collage_1x4)
    # plt.title("1 x 4 collage")

    # Create a 2 x 2 image collage in the following arrangement
    #
    # original image | red channel
    # green channel  | blue channel

    top_row = np.concatenate((img, red_image), axis=1)
    bottom_row = np.concatenate((green_image, blue_image), axis=1)
    collage_2x2 = np.concatenate((top_row, bottom_row), axis=0)
    plt.figure()
    plt.imshow(collage_2x2)
    plt.title("2 x 2 collage")

    # Save the collage as collage.png
    io.imsave("collage.png", collage_2x2)

    plt.show()
    pass
