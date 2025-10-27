import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import gray2rgb
from scipy.ndimage import maximum_filter
from tqdm import tqdm


def line_finder(orig_img, hough_img, hough_threshold):
    """
    Detect lines from Hough accumulator and overlay them on the original image.

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
    line_detected_img : ndarray
        Annotated image with detected lines.
    """

    # Ensure image is RGB for drawing
    if orig_img.ndim == 2:
        img_rgb = gray2rgb(orig_img)
    else:
        img_rgb = orig_img.copy()

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_axis_off()

    # --------------------------------------
    # TODO: START ADDING YOUR CODE HERE
    # --------------------------------------

    H, W = img_rgb.shape[:2]
    N_rho, N_theta = hough_img.shape

    # Normalize accumulator to [0, 1] for thresholding
    acc = hough_img.astype(np.float64)
    if acc.max() > 0:
        acc /= acc.max()

    # Determine threshold: if hough_threshold == 0, use adaptive threshold
    if hough_threshold > 1:
        thr = hough_threshold / 255.0
    elif hough_threshold > 0:
        thr = hough_threshold
    else:
        # Adaptive threshold: use percentile-based method
        # Focus on top votes that are significantly above noise
        thr = np.percentile(acc[acc > 0], 92) if np.any(acc > 0) else 0.0
        thr = max(thr, 0.4)  # Ensure minimum threshold to avoid noise

    # Non-maximum suppression to find local peaks
    # Use larger neighborhood to avoid duplicate nearby lines
    neighborhood = (15, 15)
    local_max = acc == maximum_filter(acc, size=neighborhood, mode='nearest')
    peak_mask = local_max & (acc >= thr)
    peak_indices = np.argwhere(peak_mask)
    
    # Sort by strength descending
    peak_indices = sorted(peak_indices, key=lambda ij: acc[ij[0], ij[1]], reverse=True)

    # Limit the number of lines to avoid clutter
    max_lines = 25
    peak_indices = peak_indices[:max_lines]

    # Parameter mapping consistent with accumulator generation
    rho_max = np.hypot(H, W)
    theta_scale = np.pi / float(N_theta)

    for rho_idx, theta_idx in peak_indices:
        theta = theta_idx * theta_scale
        # Map rho index back to value in [-rho_max, rho_max]
        if N_rho > 1:
            rho = -rho_max + (rho_idx / (N_rho - 1)) * (2.0 * rho_max)
        else:
            rho = 0.0

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        points = []
        eps = 1e-9

        # Intersect with image borders using: x*sin - y*cos + rho = 0
        # Solve for y at x=0 and x=W-1: y = (x*sin + rho) / cos
        if abs(cos_t) > eps:
            y_at_x0 = (0.0 * sin_t + rho) / cos_t
            y_at_xw = ((W - 1) * sin_t + rho) / cos_t
            if 0 <= y_at_x0 <= H - 1:
                points.append((0, y_at_x0))
            if 0 <= y_at_xw <= H - 1:
                points.append((W - 1, y_at_xw))

        # Solve for x at y=0 and y=H-1: x = (y*cos - rho) / sin
        if abs(sin_t) > eps:
            x_at_y0 = (0.0 * cos_t - rho) / sin_t
            x_at_yh = ((H - 1) * cos_t - rho) / sin_t
            if 0 <= x_at_y0 <= W - 1:
                points.append((x_at_y0, 0))
            if 0 <= x_at_yh <= W - 1:
                points.append((x_at_yh, H - 1))

        # Need two distinct points to draw the segment across the image
        if len(points) >= 2:
            # Deduplicate potential duplicates
            # Sort by x then y, take first two distinct
            unique_pts = []
            for pt in points:
                if not unique_pts or np.hypot(pt[0] - unique_pts[-1][0], pt[1] - unique_pts[-1][1]) > 1e-6:
                    unique_pts.append(pt)
            if len(unique_pts) >= 2:
                (x1, y1), (x2, y2) = unique_pts[0], unique_pts[1]
                ax.plot([x1, x2], [y1, y2], color='lime', linewidth=1.5)


    # ---------------------------
    # END YOUR CODE HERE
    # ---------------------------

    # Convert figure to image array
    fig.canvas.draw()
    line_detected_img = np.array(fig.canvas.buffer_rgba())[..., 0:3]
    plt.close(fig)

    return line_detected_img
