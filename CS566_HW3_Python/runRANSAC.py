import numpy as np
from computeHomography import compute_homography
from applyHomography import apply_homography


def run_ransac(Xs, Xd, ransac_n, eps):
    num_pts = Xs.shape[0]
    pts_id = np.arange(num_pts)
    inliers_id = np.array([])
    H = np.eye(3)  # H placeholder
    
    max_inliers = 0
    best_inliers_id = np.array([])
    best_H = np.eye(3)

    for iter in range(ransac_n):
        # ---------------------------
        # START ADDING YOUR CODE HERE
        # ---------------------------
        
        # Randomly sample 4 points (minimum needed for homography)
        if num_pts < 4:
            break
        
        sample_idx = np.random.choice(num_pts, 4, replace=False)
        sample_src = Xs[sample_idx]
        sample_dst = Xd[sample_idx]
        
        # Compute homography from the 4 sampled point pairs
        try:
            H_candidate = compute_homography(sample_src, sample_dst)
        except:
            # If homography computation fails, skip this iteration
            continue
        
        # Apply homography to all source points
        Xd_predicted = apply_homography(H_candidate, Xs)
        
        # Compute Euclidean distance between predicted and actual destination points
        distances = np.sqrt(np.sum((Xd_predicted - Xd) ** 2, axis=1))
        
        # Find inliers: points with error less than eps
        current_inliers_id = pts_id[distances < eps]
        num_inliers = len(current_inliers_id)
        
        # Keep track of the best model (most inliers)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers_id = current_inliers_id
            best_H = H_candidate

        # ---------------------------
        # END ADDING YOUR CODE HERE
        # ---------------------------

    # Recompute homography using all inliers from the best model
    if len(best_inliers_id) >= 4:
        inliers_id = best_inliers_id
        H = compute_homography(Xs[inliers_id], Xd[inliers_id])
    else:
        # If we couldn't find enough inliers, return empty
        inliers_id = best_inliers_id
        H = best_H

    return inliers_id, H
