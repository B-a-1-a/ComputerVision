import cv2
import numpy as np


def gen_sift_matches(imgs, imgd):
    # Convert to single precision grayscale
    if imgs.dtype == np.float32:
        gray_s = (cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY)
                  * 255).astype(np.uint8)
    else:
        gray_s = cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY).astype(np.uint8)

    if imgd.dtype == np.float32:
        gray_d = (cv2.cvtColor(imgd, cv2.COLOR_RGB2GRAY)
                  * 255).astype(np.uint8)
    else:
        gray_d = (cv2.cvtColor(imgd, cv2.COLOR_RGB2GRAY)).astype(np.uint8)

    # Compute SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    # Each keypoint has attributes: pt (x,y), size (scale), angle (orientation)
    Fs, Ds = sift.detectAndCompute(gray_s, None)
    Fd, Dd = sift.detectAndCompute(gray_d, None)

    # Ds and Dd are descriptors of the corresponding frames in F
    # Match descriptors using a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(Ds, Dd)
    matches = sorted(matches, key=lambda x: x.distance)

    # matches: list of DMatch objects, sorted by distance (score)
    # The two indices store the indices of Ds and Dd that match with each other

    # Extract matched points
    xs = np.array([Fs[m.queryIdx].pt for m in matches], dtype=np.float32)
    xd = np.array([Fd[m.trainIdx].pt for m in matches], dtype=np.float32)

    # xs and xd are the centers of matched frames
    # xs and xd are nx2 matrices
    return xs, xd
