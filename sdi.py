"""
Implementation of the Shadow Detection Index from:
    Liu, X. et al. (2022). Shadow Removal from UAV Images Based on Color and
    Texture Equalization Compensation of Local Homogeneous Regions.
    Remote Sensing, 14, 2616. https://doi.org/10.3390/rs14112616
Filtering in sdi_filtered() developed by the author,
    - Buildings not included as sdi is used for cast shadows
    - Vegetation is removed based on hue and saturation thresholds in HSV colour space, 
    to avoid false positives from green vegetation shadows
    - Small and large objects are removed based on pixel area thresholds, 
    to filter out noise and very large regions that are unlikely to be cast shadows of buildings.
"""

import cv2
import numpy as np
from scipy import ndimage as ndi

# --------- Default parameters ---------
_DEFAULT_OMEGA = 0.1  # weight of the 2G-B-R term
_DEFAULT_BUILDING_DILATION = 3 # building-mask dilation radius (px)
_DEFAULT_MIN_OBJECT_PX = 500 # remove objects smaller than this
_DEFAULT_MAX_OBJECT_PX = 4_000_000 # remove objects larger than this

def sdi(
    image: np.ndarray,
    omega: float = _DEFAULT_OMEGA,
) -> np.ndarray:
    """
    Detect shadow regions using the Shadow Detection Index (Liu et al. 2022)
    """
    sdi_map = _compute_sdi(image, omega)
    after_otsu, _ = _otsu_shadow_mask(sdi_map, return_thresh=True)
    mask = after_otsu.copy()
    return mask

def sdi_filtered(image: np.ndarray,
            building_mask: np.ndarray,
            omega: float = _DEFAULT_OMEGA,
            min_object_px: int = _DEFAULT_MIN_OBJECT_PX,
            max_object_px: int = _DEFAULT_MAX_OBJECT_PX,
            building_dilation_px: int = _DEFAULT_BUILDING_DILATION,
            remove_vegetation: bool = True,
            green_hue_range: tuple[int, int] = (35, 90),
            min_saturation: int = 50,
            closure: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
) -> np.ndarray:
    """Detect shadow regions with SDI and apply post-processing filters to refine the mask."""
    # Compute the initial SDI mask
    mask = sdi(image, omega=omega)

    # Erode by 5 x 5 kernel to remove small noisy regions and separate close objects.
    mask = cv2.erode(mask.astype(np.uint8), closure).astype(bool)
    
    # Remove buildings from the shadow
    mask = mask & ~building_mask.astype(bool)

    # Remove vegetation shadows based on hue and saturation thresholds in HSV color space.
    if remove_vegetation:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[..., 0]
        sat = hsv[..., 1]
        veg_mask = ((hue >= green_hue_range[0]) & (hue <= green_hue_range[1]) &
                    (sat >= min_saturation))
        mask = mask & ~veg_mask

    # Remove small and large objects based on pixel area thresholds.
    mask = _remove_small_big_objects(mask, min_object_px, max_object_px)

    # Keep only regions that are near buildings.
    mask = _filter_by_building(mask, building_mask, dilation_px=building_dilation_px)

    # Morphological closing to fill small holes and smooth edges.
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, closure).astype(bool)

    return mask

def _filter_by_building(
    mask: np.ndarray,
    building_mask: np.ndarray,
    dilation_px: int,
) -> np.ndarray:
    """
    Keep only connected components of the mask that overlap with
    a building_mask dilated by dilation_px pixels for a small buffer.
    """
    bld = (building_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
    )
    bld_dilated = cv2.dilate(bld, kernel).astype(bool)

    labeled, n_labels = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
    if n_labels == 0:
        return mask.copy()

    # For each label find whether any of its pixels fall inside bld_dilated
    flat_touch   = (labeled * bld_dilated).ravel()  # non-zero where touched

    touched = np.zeros(n_labels + 1, dtype=bool)
    for lbl in np.unique(flat_touch):
        if lbl > 0:
            touched[lbl] = True

    keep = touched[labeled] # True where label is a keeper
    return keep & mask

def _remove_small_big_objects(mask: np.ndarray, min_size: int, max_size: int) -> np.ndarray:
    """Remove connected components (8-connectivity) smaller than min_size and larger than max_size pixels."""
    labeled, n_labels = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
    if n_labels == 0:
        return mask.copy()
    sizes  = ndi.sum(mask, labeled, range(1, n_labels + 1))
    keeper = np.zeros(n_labels + 1, dtype=bool)
    keeper[0] = False
    for i, s in enumerate(sizes, start=1):
        if min_size <= s <= max_size:
            keeper[i] = True
    return keeper[labeled]

def _compute_sdi(
    image: np.ndarray,
    omega: float,
) -> np.ndarray:
    """
    Compute the raw SDI map: SDI = ω x |2G - B - R| + ε x G
    
    Input: BGR image (OpenCV convention): B=ch0, G=ch1, R=ch2. 
    (symmetric in B and R, so channel order between R and B does not affect the result.)
    """
    epsilon = 1.0 - omega

    img = image.astype(np.float32)

    B = img[..., 0]
    G = img[..., 1]
    R = img[..., 2]

    contrast_term = np.abs(2.0 * G - B - R)
    sdi_map = omega * contrast_term + epsilon * G

    return sdi_map.astype(np.float32)


def _otsu_shadow_mask(
    sdi_map: np.ndarray,
    return_thresh: bool = False,
):
    """
    Apply Otsu's method to the SDI map.
    (Shadow regions have low SDI values (darker and less green), so shadow
    pixels are below the Otsu threshold.)
    """
    sdi_min, sdi_max = sdi_map.min(), sdi_map.max()
    if sdi_max - sdi_min < 1e-6:
        mask = np.zeros(sdi_map.shape, dtype=bool)
        return (mask, 0.0) if return_thresh else mask

    sdi_u8 = np.clip(
        255.0 * (sdi_map - sdi_min) / (sdi_max - sdi_min), 0, 255
    ).astype(np.uint8)

    thresh_u8, _ = cv2.threshold(
        sdi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thresh_original = sdi_min + thresh_u8 / 255.0 * (sdi_max - sdi_min)
    mask = sdi_map < thresh_original

    if return_thresh:
        return mask, thresh_original
    return mask
