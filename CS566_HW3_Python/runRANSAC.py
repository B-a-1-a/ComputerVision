import numpy as np
from computeHomography import compute_homography
from applyHomography import apply_homography


def run_ransac(Xs, Xd, ransac_n, eps):
    num_pts = Xs.shape[0]
    pts_id = np.arange(num_pts)
    inliers_id = np.array([])
    H = np.eye(3)  # H placeholder

    for iter in range(ransac_n):
        # ---------------------------
        # START ADDING YOUR CODE HERE
        # ---------------------------

        # ---------------------------
        # END ADDING YOUR CODE HERE
        # ---------------------------
        pass  # placeholder so for loop isn't empty.

    return inliers_id, H
