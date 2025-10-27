import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature
from matplotlib import cm


def demo_python_tricks_fun():
    # Load labeled image
    threeboxes = io.imread("labeled_three_boxes.png")

    # Generate colors from the jet colormap
    n_labels = threeboxes.max() + 1
    jet_colors = cm.jet(np.linspace(0, 1, n_labels))

    # 'color' the labeled image
    rgb_img = color.label2rgb(
        threeboxes, colors=jet_colors, bg_label=0, bg_color=(0, 0, 0)
    )

    # Show the labeled image
    fig1, ax1 = plt.subplots()
    ax1.imshow(rgb_img)

    # Convert to grayscale for corner detection
    gray_img = color.rgb2gray(rgb_img)
    corners = feature.corner_peaks(
        feature.corner_harris(gray_img), min_distance=5, threshold_rel=1e-6)

    # Annotate corners
    ax1.plot(corners[:, 1], corners[:, 0], 'ws', markerfacecolor='w')
    plt.show()

    # Draw lines on the image
    loopxy = np.array([[31, 130], [31, 31], [390, 31], [390, 130], [31, 130]])
    for i in range(1, len(loopxy)):
        ax1.plot(loopxy[i-1:i+1, 0], loopxy[i-1:i+1, 1],
                 linewidth=4, color='g')

    # Save the annotated image using helper
    annotated_img = save_annotated_img(fig1)

    # Compare sizes
    print("Annotated:", annotated_img.shape)
    print("RGB img:", rgb_img.shape)

    # Display saved annotated image
    fig3, ax3 = plt.subplots()
    ax3.imshow(annotated_img)
    plt.show()


def save_annotated_img(fig):
    """
    Save a matplotlib figure with annotations as an image.
    """
    # Convert figure to image array
    fig.canvas.draw()
    buf = np.array(fig.canvas.buffer_rgba())[..., 0:3]
    return buf


if __name__ == "__main__":
    demo_python_tricks_fun()
