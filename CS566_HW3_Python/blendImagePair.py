import numpy as np
from scipy.ndimage import distance_transform_edt as bwdist


def blend_image_pair(wrapped_imgs, masks, wrapped_imgd, maskd, mode):
    Hs, Ws, Cs = wrapped_imgs.shape
    Hd, Wd, Cd = wrapped_imgd.shape

    assert (Hs == Hd) and (Ws == Wd) and (Cs == Cd)

    assert wrapped_imgs.dtype == wrapped_imgd.dtype
    assert wrapped_imgs.dtype in [np.uint8, np.float32, np.float64]

    out_img = np.zeros((Hs, Ws, Cs), dtype=np.float64)
    input_type = wrapped_imgs.dtype

    # convert to float64 to avoid overflow/underflow when multiplying with
    # the weighted mask
    wrapped_imgs = wrapped_imgs.astype(np.float64)
    wrapped_imgd = wrapped_imgd.astype(np.float64)

    binary_mask_s = masks > 0
    binary_mask_d = maskd > 0

    for c in range(Cs):
        channel_out = np.zeros((Hs, Ws), dtype=np.float64)
        S = wrapped_imgs[:, :, c]
        D = wrapped_imgd[:, :, c]
        if mode == "overlay":
            # s first, then d overwrites s wherever there is overlap.
            channel_out[binary_mask_s] = S[binary_mask_s]
            channel_out[binary_mask_d] = D[binary_mask_d]
        elif mode == "blend":
            # ---------------------------
            # ADD YOUR CODE HERE
            # ---------------------------
            #
            # you need to compute the weighted masks (for src and dest) using
            # bwdist, and use them to form the output image.
            #
            # Compute distance transform for each mask
            # bwdist computes the Euclidean distance from each pixel to the nearest zero pixel
            weight_s = bwdist(binary_mask_s)
            weight_d = bwdist(binary_mask_d)
            
            # Normalize weights so they sum to 1 in overlapping regions
            # In non-overlapping regions, the weight will be 1 for the present image
            weight_sum = weight_s + weight_d
            
            # Avoid division by zero
            weight_sum[weight_sum == 0] = 1
            
            # Normalize weights
            weight_s_norm = weight_s / weight_sum
            weight_d_norm = weight_d / weight_sum
            
            # Blend using weighted combination
            channel_out = weight_s_norm * S + weight_d_norm * D
        out_img[:, :, c] = channel_out

    # convert out_img to right type
    if input_type == np.uint8:
        out_img = np.clip(np.round(out_img), 0, 255).astype(np.uint8)
    elif input_type == np.float32:
        out_img = out_img.astype(np.float32)

    return out_img
