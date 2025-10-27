import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray
from skimage import feature
from scipy.ndimage import maximum_filter


def line_segment_finder(orig_img, hough_img, hough_threshold):
    """
    Detect line segments from Hough accumulator and draw them on the original image.

    Parameters
    ----------
    orig_img : ndarray (H, W) or (H, W, 3)
        Original grayscale or RGB image.
    hough_img : ndarray (rho_bins, theta_bins)
        Hough accumulator.
    hough_threshold : float
        Threshold above which Hough votes are considered strong.

    Returns
    -------
    cropped_line_img : ndarray
        Annotated image with detected line segments.
    """
    # Ensure image is RGB
    if orig_img.ndim == 2:
        img_rgb = gray2rgb(orig_img)
    else:
        img_rgb = orig_img.copy()

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_axis_off()

    # --------------------------------------
    # START ADDING YOUR CODE HERE
    # --------------------------------------

    H, W = img_rgb.shape[:2]
    N_rho, N_theta = hough_img.shape

    # Prepare edge map from original image for segment pruning
    if orig_img.ndim == 3:
        gray = rgb2gray(orig_img)
    else:
        gray = orig_img
    edge_img = feature.canny(gray, sigma=1.5)
    edge_y, edge_x = np.nonzero(edge_img)

    # Normalize accumulator and find local maxima
    acc = hough_img.astype(np.float64)
    if acc.max() > 0:
        acc /= acc.max()

    if hough_threshold > 1:
        thr = hough_threshold / 255.0
    elif hough_threshold > 0:
        thr = hough_threshold
    else:
        # Adaptive threshold: use percentile-based method
        thr = np.percentile(acc[acc > 0], 92) if np.any(acc > 0) else 0.0
        thr = max(thr, 0.4)  # Ensure minimum threshold

    local_max = acc == maximum_filter(acc, size=(15, 15), mode='nearest')
    peak_mask = local_max & (acc >= thr)
    peak_indices = np.argwhere(peak_mask)
    peak_indices = sorted(peak_indices, key=lambda ij: acc[ij[0], ij[1]], reverse=True)

    max_lines = 40
    peak_indices = peak_indices[:max_lines]

    rho_max = np.hypot(H, W)
    theta_scale = np.pi / float(N_theta)

    # Distance tolerance (in pixels) of edge points from a line to be considered on-segment
    distance_tolerance = .5
    min_points_on_segment = 2

    for rho_idx, theta_idx in peak_indices:
        theta = theta_idx * theta_scale
        if N_rho > 1:
            rho = -rho_max + (rho_idx / (N_rho - 1)) * (2.0 * rho_max)
        else:
            rho = 0.0

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Compute perpendicular distance of each edge point to the line
        # Using x*sin(t) - y*cos(t) + rho = 0 => d = |x*sin - y*cos + rho|
        distances = np.abs(edge_x * sin_t - edge_y * cos_t + rho)
        on_line = distances <= distance_tolerance

        if not np.any(on_line):
            continue

        xs = edge_x[on_line]
        ys = edge_y[on_line]

        # Direction vector along the line: rotate normal n=(sin, -cos) by +90° => v=(cos, sin)
        s_vals = xs * cos_t + ys * sin_t
        s_min = np.min(s_vals)
        s_max = np.max(s_vals)

        # if s_max - s_min < 1 or (xs.size < min_points_on_segment):
        #     # Too short or too few supporting points; skip
        #     continue

        # A point on the line: normal n=(sin, -cos); solve n·p + rho = 0 => p0 = -rho * n
        x0 = -rho * sin_t
        y0 =  rho * cos_t

        # Move along direction v=(cos, sin)
        x1 = x0 + s_min * (cos_t)
        y1 = y0 + s_min * (sin_t)
        x2 = x0 + s_max * (cos_t)
        y2 = y0 + s_max * (sin_t)

        # Clamp endpoints slightly inside the image
        x1 = float(np.clip(x1, 0, W - 1))
        y1 = float(np.clip(y1, 0, H - 1))
        x2 = float(np.clip(x2, 0, W - 1))
        y2 = float(np.clip(y2, 0, H - 1))

        ax.plot([x1, x2], [y1, y2], color='yellow', linewidth=2.0)

    # Convert figure to image array
    fig.canvas.draw()
    line_detected_img = np.array(fig.canvas.buffer_rgba())[..., 0:3]
    plt.close(fig)

    return line_detected_img
