# CS566 HW3 - Image Stitching and Homography

## Design Decisions and Parameters

### Challenge 1c: Image Blending (`blendImagePair.py`)

**Blending Strategy:**
- Used distance transform (`bwdist`) to create smooth weight masks for both source and destination images
- Distance transform computes Euclidean distance from each pixel to the nearest zero (invalid) pixel
- Weights are normalized so they sum to 1 in overlapping regions
- This creates a smooth transition that favors pixels further from image boundaries

**Key Implementation:**
```python
weight_s = bwdist(binary_mask_s)
weight_d = bwdist(binary_mask_d)
weight_s_norm = weight_s / (weight_s + weight_d)
weight_d_norm = weight_d / (weight_s + weight_d)
```

### Challenge 1d: Image Stitching (`stitchImg.py`)

**RANSAC Parameters:**
- `ransac_n = 100`: Maximum number of RANSAC iterations
  - Chosen as a good balance between robustness and computation time
  - Sufficient for finding accurate homographies with typical feature matches
- `ransac_eps = 3.0`: Acceptable alignment error threshold in pixels
  - Allows for minor misalignments due to lens distortion and sampling
  - Empirically found to work well for panorama applications

**Stitching Algorithm:**
1. **Reference Image Selection:** Uses the middle image as reference (placed at canvas center)
2. **Iterative Registration:** Each image is registered to the growing stitched result (not just to the reference)
   - This allows for better accumulation of transformations
   - RANSAC filters outlier matches for robust homography estimation
3. **Backward Warping:** Uses inverse homography for proper image transformation
4. **Mask Management:** Accumulates masks using logical OR to track valid regions
5. **Type Consistency:** Ensures warped images match the dtype of the stitched canvas

**Canvas Size:**
- Initial canvas: `sum(heights) × sum(widths)` 
- Provides ample space for all images regardless of rotation/translation
- Optional cropping with `bbox_crop()` can remove excess padding

**Error Handling:**
- Skips images with insufficient inliers (< 4 points) with warning message
- Prevents crashes from degenerate homographies

### Additional Notes

**Blend Mode Selection:**
- Always uses "blend" mode in `stitchImg` for seamless panoramas
- Distance-weighted blending eliminates visible seams between images

**Performance Considerations:**
- Large canvases can slow down distance transforms in blending
- For very high-resolution images, consider pre-resizing or using smaller overlap regions
- The iterative approach works well for 3-5 images; more images may benefit from graph-based optimization

### Test Results

Successfully tested on:
- Mountain panorama (3 images): Smooth wide-angle landscape
- Kitchen panorama (3 images): Interior scene with complex features
- Both show seamless blending with no visible artifacts

