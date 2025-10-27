# -------------------------------------------------------------------------
# Part 2 - Create a I <3 NY image
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform


def hw1_walkthrough2():
    # Load the image "I_Love_New_York.png" into memory
    iheartny_img = io.imread("I_Love_New_York.png")

    # we only care about the RGB channels (ignore alpha channel for this walkthrough):
    iheartny_img = iheartny_img[..., 0:3]

    # Display the image
    # plt.figure()
    # plt.imshow(iheartny_img)
    # plt.title("Original I <3 NY")

    # Load the image "nyc.png" into memory
    nyc_img = io.imread("nyc.png")

    # Resize nyc_img so the image height is 500 pixels
    height = nyc_img.shape[0]
    scale = 500 / height
    small_nyc = transform.rescale(
        nyc_img, scale, channel_axis=-1, anti_aliasing=True, preserve_range=True
    ).astype(np.uint8)

    # Resize ILoveNY so that its height is 400 pixels
    iheartny_height = iheartny_img.shape[0]
    scale = 400 / iheartny_height
    resized_iheartny = transform.rescale(
        iheartny_img, scale, channel_axis=-1, anti_aliasing=False, preserve_range=True)

    # Convert the color image into a grayscale image
    gray_iheartny_img = color.rgb2gray(resized_iheartny[..., 0:3])

    # Display the image
    # plt.figure(), plt.imshow(gray_iheartny_img, cmap="gray")
    # plt.title("Grayscale I <3 NY")

    # Convert the grayscale image into a binary mask using a threshold value
    threshold = 0.5  # reasonable default threshold in [0,1]
    # inverse so letters/logos appear white
    resized_mask = gray_iheartny_img < threshold

    # Invert the mask
    iresized_mask = ~resized_mask
    # plt.figure()
    # plt.imshow(iresized_mask, cmap="gray")
    # plt.title("Inverted Mask")

    # Note small_nyc and iresized_mask are of different height and width
    print("small_nyc shape:", small_nyc.shape)
    print("iresized_mask shape:", iresized_mask.shape)

    # No worries. Let's use the collage technique learned in Part 2 to make
    # iresized_mask with the same height and width as small_nyc
    height_diff = small_nyc.shape[0] - iresized_mask.shape[0]
    width_diff = small_nyc.shape[1] - iresized_mask.shape[1]

    # Pad left and right
    left_pad = width_diff // 2
    right_pad = width_diff - left_pad
    resized_iheartny = np.pad(
        resized_iheartny, ((0, 0), (left_pad, right_pad), (0, 0)), mode="constant")
    iresized_mask = np.pad(
        iresized_mask, ((0, 0), (left_pad, right_pad)), mode="constant")
    # plt.figure(), plt.imshow(iresized_mask, cmap="gray")
    # plt.title("Horizontally Padded Mask")

    # Pad top and bottom
    top_pad = height_diff // 2
    bottom_pad = height_diff - top_pad
    resized_iheartny = np.pad(
        resized_iheartny, ((top_pad, bottom_pad), (0, 0), (0, 0)), mode="constant")
    iresized_mask = np.pad(
        iresized_mask, ((top_pad, bottom_pad), (0, 0)), mode="constant")

    # Cast the mask to logical
    # plt.figure()
    # plt.imshow(iresized_mask, cmap="gray")
    # plt.title("Final Mask")

    # Now, let's burn the I <3 NY logo into the Manhattan scene
    love_small_nyc = small_nyc.copy()

    # Apply where mask = True

    # if you just want the red channel:
    # love_small_nyc[iresized_mask] = np.array([255, 0, 0])

    # or, if you want all channels:
    channel_mask = np.repeat(iresized_mask[:, :, np.newaxis], 3, axis=2)
    love_small_nyc[channel_mask] = np.clip(resized_iheartny, 0, 255).astype(np.uint8)[channel_mask]

    plt.figure()
    plt.imshow(love_small_nyc)
    plt.title("I <3 NY on Manhattan")

    # Save the collage as output_nyc.png
    io.imsave("output_nyc.png", love_small_nyc.astype(np.uint8))

    plt.show()
