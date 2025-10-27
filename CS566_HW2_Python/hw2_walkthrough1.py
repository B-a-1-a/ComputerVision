import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, img_as_ubyte
from scipy.ndimage import convolve
from skimage.filters import gaussian, sobel
from skimage.feature import canny


def hw2_walkthrough1():
    # -----------------------
    # Image processing: convolution, Gaussian smoothing
    # -----------------------

    # Read image
    img = io.imread('flower.png')

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Apply Gaussian blur with three different sigmas
    sigma_list = [6, 12, 24]
    for i, sigma in enumerate(sigma_list):
        # Kernel size rule of thumb: k ~ 2*pi*sigma
        k = int(np.ceil(2 * np.pi * sigma))

        # Apply Gaussian filter
        blur_img = gaussian(img, sigma=sigma, truncate=(
            k/(2*sigma)), channel_axis=2)

        axes[i+1].imshow(blur_img)
        axes[i+1].set_title(f'σ = {sigma}')
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig('blur_flowers.png')
    plt.close(fig)

    # -----------------------
    # Edge detection
    # -----------------------

    # Read color image
    img = io.imread('hello.png')

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    axes[0].imshow(img)
    axes[0].set_title('Color Image')
    axes[0].axis('off')

    # Convert to grayscale
    gray_img = color.rgb2gray(img)
    axes[1].imshow(gray_img, cmap='gray')
    axes[1].set_title('Grayscale Image')
    axes[1].axis('off')

    # Sobel edge detection
    sobel_response = sobel(gray_img)
    thresh_sobel = filters.threshold_otsu(sobel_response)
    edge_sobel = sobel_response > thresh_sobel
    axes[2].imshow(edge_sobel, cmap='gray')
    axes[2].set_title('Sobel Edge Detection')
    axes[2].axis('off')

    # Canny edge detection
    edge_canny = canny(gray_img, sigma=1.0)
    axes[3].imshow(edge_canny, cmap='gray')
    axes[3].set_title('Canny Edge Detection')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('hello_edges.png')
    plt.close(fig)
