import numpy as np
from applyHomography import apply_homography
from scipy.ndimage import map_coordinates


def backward_warp_img(src_img, resultToSrc_H, dest_canvas_width_height):
    src_height = src_img.shape[0]
    src_width = src_img.shape[1]
    src_channels = src_img.shape[2]
    dest_width = dest_canvas_width_height[0]
    dest_height = dest_canvas_width_height[1]

    result_img = np.zeros((dest_height, dest_width, src_channels))
    mask = np.zeros((dest_height, dest_width), dtype=bool)

    # this is the overall region covered by result_img
    dest_X, dest_Y = np.meshgrid(np.arange(1, dest_width + 1),
                                 np.arange(1, dest_height + 1))

    # map result_img region to src_img coordinate system using the given homography
    src_pts = apply_homography(resultToSrc_H, np.column_stack(
        [dest_X.ravel(), dest_Y.ravel()]))
    src_X = src_pts[:, 0].reshape(dest_height, dest_width)
    src_Y = src_pts[:, 1].reshape(dest_height, dest_width)

    # ---------------------------
    # START ADDING YOUR CODE HERE
    # ---------------------------

    # Set 'mask' to the correct values based on src_pts.
    # Mask is True where the source coordinates are within the bounds of the source image
    # (using 1-indexed coordinates to match the meshgrid)
    mask = (src_X >= 1) & (src_X <= src_width) & \
           (src_Y >= 1) & (src_Y <= src_height)

    # fill the right region in 'result_img' with the src_img
    # Use map_coordinates for bilinear interpolation
    # map_coordinates expects (row, col) = (Y, X) order and 0-indexed coordinates
    for c in range(src_channels):
        result_img[:, :, c] = map_coordinates(
            src_img[:, :, c],
            [src_Y - 1, src_X - 1],  # Convert from 1-indexed to 0-indexed
            order=1,  # bilinear interpolation
            mode='constant',
            cval=0
        )
    
    # Apply mask to zero out pixels outside the valid region
    result_img = result_img * mask[:, :, np.newaxis]

    # ---------------------------
    # END YOUR CODE HERE
    # ---------------------------

    return mask, result_img
