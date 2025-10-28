import numpy as np
from genSIFTMatches import gen_sift_matches
from runRANSAC import run_ransac
from backwardWarpImg import backward_warp_img
from blendImagePair import blend_image_pair
from bboxCrop import bbox_crop
import matplotlib.pyplot as plt


def stitch_img(*args):
    # GENERAL NOTE: Feel free to change all of this file, not just the
    #               "ADD YOUR CODE HERE" sections. We're just trying to help
    #               get you started.

    # The code below makes sure there is a very large canvas for us to put
    # the stitched image in. It's height is twice the sum of the heights of
    # the input images, and its width is twice the sum of their widths.
    #
    # This makes the image really large, so you might want to crop the
    # blank borders at the end, using the helper function bbox_crop.
    H_stitched = sum([img.shape[0] for img in args])
    W_stitched = sum([img.shape[1] for img in args])

    # Images should be all grayscale or all colour
    assert max([img.shape[2] for img in args]) == min(
        [img.shape[2] for img in args])
    C_stitched = args[0].shape[2]

    stitched_img = np.zeros(
        (H_stitched, W_stitched, C_stitched), dtype=args[0].dtype)

    # NOTE: The scaffolding code given below assumes that the reference
    # image is the "middle" image in the image sequence passed in through
    # 'varargin'. So if you call this function like:
    #       stitchImg(img_l, img_c, img_r)
    # for images taken left-to-right in the sequence [img_l, img_c, img_r],
    # this code will assume img_c is the reference and it covers the middle
    # of the canvas.
    #
    # If you'd like to do something else, you will have to change the
    # scaffolding code in addition to the new code that you add.
    num_imgs = len(args)
    middle_idx = round((num_imgs + 1) / 2)
    # NOTE: you can put a different value here if you want!
    ref_idx = middle_idx - 1  # adjust for 0-based indexing

    # paste the reference image into the output canvas.
    ref_img = args[ref_idx]
    H_ref, W_ref, _ = ref_img.shape
    ref_start_x = 0 + (W_stitched - W_ref) // 2
    ref_start_y = 0 + (H_stitched - H_ref) // 2

    stitched_img[ref_start_y: ref_start_y + H_ref,
                 ref_start_x: ref_start_x + W_ref,
                 :] = ref_img

    stitch_mask = np.zeros((H_stitched, W_stitched), dtype=bool)
    stitch_mask[ref_start_y: ref_start_y + H_ref,
                ref_start_x: ref_start_x + W_ref] = True

    for n in range(num_imgs):
        if n == ref_idx:
            continue
        img_n = args[n]

        kp_stitched, kp_n = gen_sift_matches(stitched_img, img_n)

        # ---------------------------------------
        # ADD YOUR CODE HERE
        # ---------------------------------------
        # Blend img_n into stitched_img, after finding the right homography
        # to register it, and warping it with the reverse transformation
        # (backward warp).
        # Use RANSAC to avoid problems caused by outliers.
        #
        # Run RANSAC to find homography
        ransac_n = 100  # Max number of iterations
        ransac_eps = 3.0  # Acceptable alignment error in pixels
        
        # Run RANSAC to find inliers and compute robust homography
        # kp_n are source points, kp_stitched are destination points
        inliers_id, H_n_to_stitched = run_ransac(kp_n, kp_stitched, ransac_n, ransac_eps)
        
        if len(inliers_id) < 4:
            print(f"Warning: Not enough inliers found for image {n}. Skipping.")
            continue
        
        # Warp img_n into the stitched coordinate system
        # We need the inverse homography for backward warping
        H_stitched_to_n = np.linalg.inv(H_n_to_stitched)
        mask_n, warped_img_n = backward_warp_img(
            img_n, H_stitched_to_n, (W_stitched, H_stitched))
        
        # Ensure warped image has the same dtype as stitched_img
        warped_img_n = warped_img_n.astype(stitched_img.dtype)
        
        # Blend the warped image with the current stitched image
        stitched_img = blend_image_pair(
            stitched_img, stitch_mask, warped_img_n, mask_n, mode="blend")
        
        # Update the stitch mask to include the new region
        stitch_mask = np.logical_or(stitch_mask, mask_n)

        # ---------------------------------------
        # END ADD YOUR CODE HERE
        # ---------------------------------------

    # OPTIONAL: remove excess padding from the output
    # stitched_img = bbox_crop(stitched_img)

    return stitched_img
