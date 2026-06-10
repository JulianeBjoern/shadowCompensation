import numpy as np
import cv2
from PIL import Image
from itertools import combinations
from scipy.ndimage import convolve, binary_closing

# agha(), silva(), sdi() and sdi_nogreen() are the main functions for shadow detection.

def agha(image: np.ndarray) -> np.ndarray:

    rgb = _to_rgb_float(image)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    rm1 = _ratio_map_1(R, G, B)
    rm1_mod = _modified_ratio_map(rm1)
    bin1 = _otsu_threshold(rm1_mod)

    rm2 = _ratio_map_2(R, G, B)
    rm2_mod = _modified_ratio_map(rm2)
    bin2 = _otsu_threshold(rm2_mod)

    shadow_mask = bin1 & bin2

    return shadow_mask

def _to_rgb_float(image: np.ndarray) -> np.ndarray:

    img = image.astype(np.float32)

    if img.ndim == 3 and img.shape[2] == 3:
        img = img[..., ::-1]
    return img

def _intensity(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:

    return (R + G + B) / 3.0

def _ratio_map_1(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:

    I = _intensity(R, G, B)
    numerator = 2.0 * B + 1.0
    denominator = (R + G + 1.0) * (I + 1.0)
    rm = np.round(numerator / denominator * 255.0)
    return np.clip(rm, 0, 255)

_W_R = 0.26
_W_G = 0.34
_W_B = 0.40

def _proposed_saturation(R: np.ndarray, G: np.ndarray,
                         B: np.ndarray) -> np.ndarray:

    r, g, b = R / 255.0, G / 255.0, B / 255.0
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    denom = _W_R * r + _W_G * g + _W_B * b + 1.0
    S = 1.0 - (2.0 * Y) / denom
    return np.clip(S, 0.0, 1.0)

def _ratio_map_2(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:

    S = _proposed_saturation(R, G, B)
    I = _intensity(R, G, B)
    rm = np.round((S * 255.0) / (I + 1.0))
    return np.clip(rm, 0, 255)

def _modified_ratio_map(rm: np.ndarray, ps: float = 0.95) -> np.ndarray:

    rm = rm.astype(np.float32)
    flat = rm.flatten()
    hist, bins = np.histogram(flat, bins=256, range=(0, 255))
    total = flat.size
    prob = hist / total

    cumulative = np.cumsum(prob)
    ts_idx = np.searchsorted(cumulative, ps)
    Ts = float(np.clip(ts_idx, 1, 254))

    i_vals = np.arange(len(prob), dtype=np.float32)
    mask_below = i_vals < Ts
    sigma_sq = float(np.sum(prob[mask_below] * (i_vals[mask_below] - Ts) ** 2))
    sigma_sq = max(sigma_sq, 1e-6)

    rm_mod = np.where(
        rm < Ts,
        np.exp(-((rm - Ts) ** 2) / (4.0 * sigma_sq)) * 255.0,
        255.0
    )
    return np.clip(rm_mod, 0, 255).astype(np.float32)

def _otsu_threshold(rm_mod: np.ndarray) -> np.ndarray:

    img_u8 = np.clip(rm_mod, 0, 255).astype(np.uint8)
    thresh, _ = cv2.threshold(img_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_u8 >= thresh

def silva(image, nbins=256, noise_structure=12, classes=3, dilate_structure=0):

    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    RGB = np.array(image)

    RGBtoCIEXYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])

    CIEXYZ = RGB @ RGBtoCIEXYZ.T
    [X, Y, Z] = [CIEXYZ[..., 0], CIEXYZ[..., 1], CIEXYZ[..., 2]]

    [Xn, Yn, Zn] = [95.047, 100.000, 108.883]

    def L_func(Y):
        return np.where(Y / Yn > 0.008856, 116 * (Y / Yn) ** (1/3) - 16, 903.3 * (Y / Yn))
    def f(x):
        return np.where(x > 0.008856, x ** (1/3), 7.787 * x + 16/116)

    L = L_func(Y)
    a = 500 * (f(X / Xn) - f(Y / Yn))
    b = 200 * (f(Y / Yn) - f(Z / Zn))

    C = np.sqrt(a ** 2 + b ** 2)
    h = np.arctan2(b, a) * 180 / np.pi
    h[h < 0] += 360
    h[h >= 360] -= 360

    L = (L - L.min()) / (L.max() - L.min())
    h = (h - h.min()) / (h.max() - h.min())

    Sr = (h + 1) / (L + 1)

    SrLog = np.log1p(Sr + 1)

    Bf = np.ones((5, 5)) / 25
    SrLogBlurred = convolve(SrLog, Bf, mode='reflect')

    def multi_otsu_thresholds(image, classes=classes, nbins=nbins):

        pixels = image.ravel()

        hist, bin_edges = np.histogram(pixels, bins=nbins, density=True)
        p = hist

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        mu_T = np.sum(bin_centers * p)

        best_sigma = -np.inf
        best_thresholds = None

        for indices in combinations(range(1, nbins - 1), classes - 1):
            thresholds = (0,) + indices + (nbins,)
            sigma_B = 0.0

            for k in range(classes):
                i0, i1 = thresholds[k], thresholds[k + 1]
                w_k = np.sum(p[i0:i1])
                if w_k == 0:
                    break
                mu_k = np.sum(bin_centers[i0:i1] * p[i0:i1]) / w_k
                sigma_B += w_k * (mu_k - mu_T) ** 2
            else:
                if sigma_B > best_sigma:
                    best_sigma = sigma_B
                    best_thresholds = indices

        return [bin_centers[t] for t in best_thresholds]

    thresholds = multi_otsu_thresholds(SrLogBlurred)

    T = max(thresholds)
    shadow_mask_noisy = SrLogBlurred > T

    if dilate_structure > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_structure, dilate_structure))
        shadow_mask_noisy = cv2.dilate(shadow_mask_noisy.astype(np.uint8), kernel).astype(bool)

    A = shadow_mask_noisy.astype(np.uint8)
    B = np.ones((noise_structure, noise_structure), dtype=bool)
    shadow_mask = binary_closing(A, structure=B)

    return shadow_mask

_DEFAULT_OMEGA = 0.1

def sdi(image: np.ndarray, *, omega: float = _DEFAULT_OMEGA,
        epsilon: float | None = None) -> np.ndarray:

    sdi_map = _compute_sdi(image, omega, epsilon)
    after_otsu, otsu_thresh = _otsu_shadow_mask(sdi_map, return_thresh=True)
    mask = after_otsu.copy()
    return mask

def sdi_nogreen(image: np.ndarray,
                omega: float = _DEFAULT_OMEGA,
                remove_vegetation: bool = True,
                green_hue_range: tuple[int, int] = (35, 90),
                min_saturation: int = 50,
                ) -> np.ndarray:

    mask = sdi(image, omega=omega, epsilon=1.0 - omega)

    if remove_vegetation:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[..., 0]
        sat = hsv[..., 1]
        veg_mask = ((hue >= green_hue_range[0]) & (hue <= green_hue_range[1]) &
                    (sat >= min_saturation))
        mask = mask & ~veg_mask

    return mask

def _compute_sdi(image: np.ndarray, omega: float,
                 epsilon: float | None) -> np.ndarray:

    if epsilon is None:
        epsilon = 1.0 - omega

    img = image.astype(np.float32)

    B = img[..., 0]
    G = img[..., 1]
    R = img[..., 2]

    contrast_term = np.abs(2.0 * G - B - R)
    sdi_map = omega * contrast_term + epsilon * G

    return sdi_map.astype(np.float32)

def _otsu_shadow_mask(sdi_map: np.ndarray, return_thresh: bool = False):

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