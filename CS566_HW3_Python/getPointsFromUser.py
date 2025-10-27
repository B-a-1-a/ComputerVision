import matplotlib.pyplot as plt
import numpy as np


def get_points_from_user(img, npts, message=None):
    fig, ax = plt.subplots()
    ax.imshow(img)
    if message is not None:
        fig.canvas.manager.set_window_title(
            message)  # Set the figure window title

    # ginput allows user to click 'npts' points on the image
    pts = plt.ginput(npts, timeout=-1)
    plt.close(fig)

    return np.array(pts)
