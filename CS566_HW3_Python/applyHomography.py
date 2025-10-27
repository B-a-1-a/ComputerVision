import numpy as np


def apply_homography(H_3x3, src_pts_nx2):

    n = src_pts_nx2.shape[0]
    src_pts_nx3 = np.hstack([src_pts_nx2, np.ones((n, 1))])  # make homogeneous
    dest_pts_nx3 = (H_3x3 @ src_pts_nx3.T).T
    dest_pts_nx2 = dest_pts_nx3[:, :2] / dest_pts_nx3[:, 2, np.newaxis]
    return dest_pts_nx2
