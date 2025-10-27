#!/usr/bin/env python3
"""
Helper script to interactively select correspondence points for Challenge 1a.
Run this once to determine the coordinates, then hard-code them in runHw3.py
"""

import cv2
import numpy as np
from getPointsFromUser import get_points_from_user

# Load images
portrait_img = cv2.imread("portrait_small.png")
bg_img = cv2.imread("Osaka.png")

print("Portrait image shape:", portrait_img.shape)
print("Osaka image shape:", bg_img.shape)

portrait_h, portrait_w = portrait_img.shape[:2]

# Get portrait corners (clockwise from top-left)
print("\nPortrait corners (in image coordinates):")
print(f"  Top-left: (1, 1)")
print(f"  Top-right: ({portrait_w}, 1)")
print(f"  Bottom-right: ({portrait_w}, {portrait_h})")
print(f"  Bottom-left: (1, {portrait_h})")

# Use these as portrait points (corners of the portrait image)
portrait_pts = np.array([
    [1, 1],                    # top-left
    [portrait_w, 1],           # top-right
    [portrait_w, portrait_h],  # bottom-right
    [1, portrait_h]            # bottom-left
], dtype=np.float32)

print("\n" + "="*60)
print("Now click 4 CORRESPONDING points on the Osaka billboard")
print("Click in the SAME ORDER:")
print("  1. Top-left corner of billboard")
print("  2. Top-right corner of billboard")
print("  3. Bottom-right corner of billboard")
print("  4. Bottom-left corner of billboard")
print("="*60)

# Get corresponding points on the billboard
bg_pts = get_points_from_user(bg_img, 4, "Click 4 corners of the billboard (clockwise from top-left)")

print("\nSelected billboard points:")
for i, (x, y) in enumerate(bg_pts):
    corners = ["top-left", "top-right", "bottom-right", "bottom-left"]
    print(f"  {corners[i]}: ({x:.1f}, {y:.1f})")

print("\n" + "="*60)
print("Copy these values into runHw3.py challenge1a():")
print("="*60)
print("\nportrait_pts = np.array([")
for i, pt in enumerate(portrait_pts):
    print(f"    [{pt[0]:.1f}, {pt[1]:.1f}],  # {['top-left', 'top-right', 'bottom-right', 'bottom-left'][i]}")
print("])")
print("\nbg_pts = np.array([")
for i, pt in enumerate(bg_pts):
    print(f"    [{pt[0]:.1f}, {pt[1]:.1f}],  # {['top-left', 'top-right', 'bottom-right', 'bottom-left'][i]}")
print("])")

