import numpy as np


def compute_homography(src_pts_nx2, dest_pts_nx2):
    # H * [xs; ys; 1] = [xd; yd; 1] (up to scale)
    # This is equivalent to:
    #   h11 * xs + h12 * ys + h13 = xd
    #   h21 * xs + h22 * ys + h23 = yd
    #   h31 * xs + h32 * ys + h33 = 1
    #
    #   Subtract xd times the 3rd equation from the first:
    #   h11*xs + h12*ys + h13 - h31*xs*xd - h32*ys*xd - h33*xd = 0
    #
    #   and subtract yd times the 3rd equation from the second:
    #   h21*xs + h22*ys + h23 - h31*xs*yd - h32*ys*yd - h33*yd = 0
    #
    #   We fix h33 = 1.
    #
    #   So, if we flatten H into a vector:
    #       [h11 h12 h13 h21 h22 h23 h31 h32 1],
    #   we have the equation A h = 0 if we take A to be the matrix coded
    #   below.
    #   We take its minimum-singular value singular vector to be h, and
    #   convert it back to 3 x 3.

    n = src_pts_nx2.shape[0]
    A = np.zeros((n * 2, 9))

    for i in range(n):
        xs, ys = src_pts_nx2[i, 0], src_pts_nx2[i, 1]
        xd, yd = dest_pts_nx2[i, 0], dest_pts_nx2[i, 1]
        A[2*i, :] = [xs, ys, 1, 0, 0, 0, -xd*xs, -xd*ys, -xd]
        A[2*i+1, :] = [0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys, -yd]

    # Eigen decomposition of A^T A
    eigvals, eigvecs = np.linalg.eig(A.T @ A)
    idx = np.argmin(eigvals)
    h = eigvecs[:, idx]

    H_3x3 = h.reshape((3, 3))
    return H_3x3
