from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import label, find_objects, distance_transform_edt

#%% --------- Self-cast shadows ---------
def reinhard_building(img_lab_comp, shadow_walls_bid, mean_sun, std_sun):
    """
    In-place Reinhard transfer for self-shadowed wall pixels of one building.
    """
    # Get the shadowed wall pixels for this building
    shadow_pixels = img_lab_comp[shadow_walls_bid]
    if shadow_pixels.size == 0:
        return
    # Compute mean and std of the shadow pixels, and apply Reinhard transfer to match the sunlit stats.
    mean_shadow = shadow_pixels.mean(axis=0)
    std_shadow  = shadow_pixels.std(axis=0) + 1e-6
    corrected   = (shadow_pixels - mean_shadow) * (std_sun / std_shadow) + mean_sun
    corrected   = np.clip(corrected,
                          mean_sun - 1.5 * std_sun,
                          mean_sun + 1.5 * std_sun)
    img_lab_comp[shadow_walls_bid] = corrected

#%% --------- Cast shadows ---------
def surround_statistics(img_lab, region_crop, shadow_ground_crop,
                        mask_building_union_crop,
                        inner_size=7, outer_size=7, min_pixels=50):
    """
    Colour statistics from the sunlit ring surrounding a shadow region.
    """
    # Ensure outer_size is at least inner_size + 3 to have a meaningful surround ring
    outer_size = max(outer_size, inner_size + 3)
    if not region_crop.any():
        return None
    # Define inner and outer dilations of the region to get a surround ring, excluding shadow ground and building union areas.
    region_u8    = region_crop.astype(np.uint8)
    inner        = cv2.dilate(region_u8, np.ones((inner_size, inner_size), np.uint8)).astype(bool)
    outer        = cv2.dilate(region_u8, np.ones((outer_size, outer_size), np.uint8)).astype(bool)
    surround     = outer & ~inner & ~shadow_ground_crop & ~mask_building_union_crop
    # Check if we have enough surround pixels to compute statistics
    if surround.sum() < min_pixels:
        return None
    # Get the LAB pixel values of the surround and compute mean and std
    pixels = img_lab[surround]
    return np.mean(pixels, axis=0), np.std(pixels, axis=0) + 1e-6, surround

def reinhard_cast(img_lab_comp, shadow_cast, building_mask, touching_mask,
                  max_dilate=7, max_penumbra=25, min_region_size=200):
    """In-place Reinhard transfer + penumbra blending for cast-shadow regions."""
    # Label connected shadow regions and iterate over them
    shadow_labels, _ = label(shadow_cast)
    region_slices    = find_objects(shadow_labels)
    for region_label, slc in enumerate(region_slices, start=1):
        if slc is None:
            continue
        if not touching_mask[slc].any():
            continue
        # Check if the region is large enough to process
        region_crop = shadow_labels[slc] == region_label
        if region_crop.sum() < min_region_size:
            continue
        # Get the corresponding crops of the shadow, building union mask, and the image LAB values for this region
        shadow_crop   = shadow_cast[slc]
        building_crop = building_mask[slc]
        img_crop      = img_lab_comp[slc]
        # Compute the surround statistics for this shadow region, excluding building areas and shadow ground
        stats = surround_statistics(img_crop, region_crop, shadow_crop, building_crop)
        if stats is None:
            continue
        mean_surround, std_surround, surround = stats

        # Apply Reinhard compensation
        shadow_pixels = img_crop[region_crop]
        mean_shadow   = shadow_pixels.mean(axis=0)
        std_shadow    = shadow_pixels.std(axis=0) + 1e-6
        corrected     = (shadow_pixels - mean_shadow) * (std_surround / std_shadow) + mean_surround
        corrected     = np.clip(corrected,
                                mean_surround - 1.5 * std_surround,
                                mean_surround + 1.5 * std_surround)
        img_crop[region_crop] = corrected   # written back via view

        # Define the penumbra as the dilated region minus the original region and building areas
        penumbra = (
            cv2.dilate(region_crop.astype(np.uint8),
                       np.ones((max_dilate, max_dilate), np.uint8)).astype(bool)
            & ~region_crop
            & ~building_crop
        )
        dist, _ = distance_transform_edt(~surround, return_indices=True)

        # For each penumbra pixel, if it's within max_penumbra distance to the surround, replace it with a random sample from the surround pixels.
        for y, x in zip(*np.where(penumbra)):
            if dist[y, x] > max_penumbra:
                continue
            y0 = max(0, y - max_penumbra)
            y1 = min(img_crop.shape[0], y + max_penumbra + 1)
            x0 = max(0, x - max_penumbra)
            x1 = min(img_crop.shape[1], x + max_penumbra + 1)
            local = img_crop[y0:y1, x0:x1][surround[y0:y1, x0:x1]]
            if len(local):
                img_crop[y, x] = local[np.random.randint(len(local))]