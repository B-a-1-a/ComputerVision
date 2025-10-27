import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def generate_hough_accumulator(img, theta_num_bins, rho_num_bins):
    """
    Generate a Hough accumulator array for an edge image.

    Parameters
    ----------
    img : ndarray (H, W)
        Edge image (nonzero pixels are treated as edges).
    theta_num_bins : int
        Number of bins for theta.
    rho_num_bins : int
        Number of bins for rho.

    Returns
    -------
    hough_img : ndarray (rho_num_bins, theta_num_bins)
        Hough accumulator normalized to 0-255.
    """

    # ---------------------------
    # Hough voting using: x*sin(theta) - y*cos(theta) + rho = 0
    # Rearranged => rho = y*cos(theta) - x*sin(theta)
    # ---------------------------

    # Ensure input is a clean boolean edge map
    edge_img = img > 0.5

    H, W = edge_img.shape

    # Theta in [0, pi)
    thetas = np.linspace(0.0, np.pi, theta_num_bins, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Rho range based on image diagonal
    rho_max = np.hypot(H, W)

    # Accumulator
    accumulator = np.zeros((rho_num_bins, theta_num_bins), dtype=np.float64)

    # Edge coordinates (y, x)
    ys, xs = np.nonzero(edge_img)
    if ys.size == 0:
        return accumulator

    # Vectorized voting: for each edge point, compute rho for all thetas
    # rho_vals shape: (num_edges, theta_num_bins)
    # rho = y*cos(theta) - x*sin(theta)
    rho_vals = (ys[:, None] * cos_t[None, :]) - (xs[:, None] * sin_t[None, :])

    # Map rho in [-rho_max, rho_max] to index in [0, rho_num_bins-1]
    rho_idx = np.round((rho_vals + rho_max) * (rho_num_bins - 1) / (2.0 * rho_max)).astype(np.int64)

    # Clip indices to be safe (numerical rounding)
    np.clip(rho_idx, 0, rho_num_bins - 1, out=rho_idx)

    # Theta indices are simply 0..theta_num_bins-1, broadcasted per edge
    theta_idx = np.arange(theta_num_bins, dtype=np.int64)[None, :]
    theta_idx = np.broadcast_to(theta_idx, rho_idx.shape)

    # Flatten and accumulate
    flat_rho = rho_idx.ravel()
    flat_theta = theta_idx.ravel()
    np.add.at(accumulator, (flat_rho, flat_theta), 1)

    # Apply slight Gaussian smoothing to make peaks more robust
    # This helps reduce noise and concentrates votes from collinear points
    accumulator = gaussian_filter(accumulator, sigma=1.0)

    # Normalize to 0..255 and clip to ensure range
    max_val = accumulator.max()
    if max_val > 0:
        accumulator = accumulator * (255.0 / max_val)
        accumulator = np.clip(accumulator, 0.0, 255.0)

    return accumulator
