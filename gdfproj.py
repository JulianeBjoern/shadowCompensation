#%%
from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pysolar.solar import get_altitude, get_azimuth

from skimage.draw import polygon as raster_polygon
from skimage.morphology import remove_small_objects

from scipy.ndimage import binary_closing

from numba import njit
from shapely.geometry import MultiPoint
from shapely.strtree import STRtree

#%% Get ground elevation from GDF
def get_ground_z(gdf):
    all_z = np.concatenate([
        np.asarray(geom.exterior.coords)[:, 2]
        for geom in gdf.geometry.values
    ])
    return float(all_z.min())

#%% ------------- Project GDF geometries to camera view -------------
def project_points_with_depth(coords, cam) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3-D world coordinates to 2-D image coordinates and return depth along the camera ray.
    """
    # Coordinate system: X right, Y forward, Z up
    X = coords[:, 0]
    Y = coords[:, 1]
    Z = coords[:, 2]
    # Camera position in world coordinates
    dX = X - cam.Xc
    dY = Y - cam.Yc
    dZ = Z - cam.Zc
    # Focal length and principal point
    f = cam.f_mm / cam.pixel_size
    x0 = cam.sensor_cols * 0.5 + cam.ppo_x / cam.pixel_size
    y0 = cam.sensor_rows * 0.5 + cam.ppo_y / cam.pixel_size
    # Depth along camera ray (negative in front of camera, positive behind)
    raw_depth = cam.m31 * dX + cam.m32 * dY + cam.m33 * dZ
    forward_depth = -raw_depth
    # Image coordinates (with y flipped to match image origin at top-left)
    xa = x0 - f * (cam.m11 * dX + cam.m12 * dY + cam.m13 * dZ) / raw_depth
    ya = y0 - f * (cam.m21 * dX + cam.m22 * dY + cam.m23 * dZ) / raw_depth
    x = xa
    y = -(ya - cam.sensor_rows)
    # Return pixel coordinates and forward-facing depth
    return np.column_stack([x, y]), forward_depth

def polygon_to_pixel_indices(img_coords, height, width) -> tuple[np.ndarray, np.ndarray]:
    """Convert projected polygon coordinates to pixel indices, handling out-of-bounds gracefully."""
    return raster_polygon(img_coords[:, 1], img_coords[:, 0], (height, width))

#%% ------------- Crop GDF to camera footprint -------------
def _unproject_pixel_to_ground(px, py, cam, z_ground: float) -> tuple[float, float] | None:
    """
    Unproject an image pixel to the ground plane (z = z_ground). Returns (X, Y) or None if no intersection.
    """
    # Focal length and principal point
    f  = cam.f_mm / cam.pixel_size
    x0 = cam.sensor_cols * 0.5 + cam.ppo_x / cam.pixel_size
    y0 = cam.sensor_rows * 0.5 + cam.ppo_y / cam.pixel_size
    # Image coordinates (undo y flip)
    ya = -(py - cam.sensor_rows)
    xa = px

    # Normalised image-plane coords
    dx_img = -(xa - x0) / f
    dy_img = -(ya - y0) / f

    # Camera rotation matrix
    R = np.array([
        [cam.m11, cam.m12, cam.m13],
        [cam.m21, cam.m22, cam.m23],
        [cam.m31, cam.m32, cam.m33],
    ])
    # Ray direction in camera coordinates
    d_cam = np.array([dx_img, dy_img, 1.0])
    # Ray direction in world coordinates
    d_world = R.T @ d_cam

    # Intersect ray with ground plane z = z_ground
    dz = d_world[2]
    if abs(dz) < 1e-10: # ray parallel to ground
        return None
    t = (z_ground - cam.Zc) / dz
    if t < 0: # ground behind camera
        return None

    # Compute intersection point
    X = cam.Xc + t * d_world[0]
    Y = cam.Yc + t * d_world[1]
    return X, Y


def build_camera_ground_footprint(cam, z_ground: float,
                                  margin_m: float = 10.0) -> MultiPoint | None:
    """
    Return a Shapely Polygon that is the camera's field-of-view projected onto z_ground

    For highly oblique cameras the four corner rays may not all intersect the ground plane (e.g. rays pointing above the horizon).  
    We fall back to the subset that does. 
    If fewer than 3 rays hit the ground we return None.
    """
    H, W = int(cam.sensor_rows), int(cam.sensor_cols)

    # Sample corners + edge midpoints (matters for wide-FOV sensors)
    sample_pixels = [
        (0,   0),   (W,   0),   (W,   H),   (0,   H),   # corners
        (W/2, 0),   (W,   H/2), (W/2, H),   (0,   H/2), # edge midpoints
    ]

    # Unproject to ground plane and collect valid points
    pts = []
    for px, py in sample_pixels:
        r = _unproject_pixel_to_ground(px, py, cam, z_ground)
        if r is not None:
            pts.append(r)

    if len(pts) < 3:
        return None  # camera almost vertical or ground behind sensor
    
    # Convex hull of valid points is the footprint. Buffer by margin_m to be safe.
    footprint = MultiPoint(pts).convex_hull
    if margin_m > 0:
        footprint = footprint.buffer(margin_m)

    return footprint

def cull_gdf_to_camera(gdf, cam, z_ground: float | None = None, margin_m: float = 10.0,
                       _footprint_cache: dict | None = None) -> gpd.GeoDataFrame:
    """
    Return the subset of the gdf (GeoDataFrame) whose geometries intersect the camera's ground-plane footprint.

    gdf: GeoDataFrame with 3-D polygon geometries
    cam: camera metadata fetched from Skraafotos
    z_ground: ground-plane elevation; inferred from gdf if None
    margin_m: buffer around the footprint in world units (metres)
    _footprint_cache: optional dict keyed by cam.datetime for re-use across tiles
    """
    if z_ground is None:
        z_ground = get_ground_z(gdf)

    # Cache the footprint so we don't recompute it for every tile
    key = getattr(cam, "datetime", id(cam))
    if _footprint_cache is not None and key in _footprint_cache:
        footprint = _footprint_cache[key]
    else:
        footprint = build_camera_ground_footprint(cam, z_ground, margin_m)
        if _footprint_cache is not None:
            _footprint_cache[key] = footprint

    if footprint is None:
        # Return the full GDF unchanged if we can't compute a footprint
        return gdf

    # bbox pre-filter via STRtree (very fast)
    tree = STRtree(gdf.geometry.values)
    candidate_idx = tree.query(footprint) # returns integer positional indices

    if len(candidate_idx) == 0:
        return gdf.iloc[0:0] # empty, same schema

    candidates = gdf.iloc[candidate_idx]

    # Use the 2-D footprint against 3-D polygons. Shapely compares (X, Y) only.
    mask = candidates.geometry.intersects(footprint)
    return candidates[mask]

# ------------- Shadow wall projection with visibility mask -------------
def parse_utc_datetime(dt_str: str) -> datetime:
    # Pysolar returns ISO format with 'Z' suffix for UTC, but datetime.fromisoformat doesn't handle 'Z', so we replace it with '+00:00' to indicate UTC offset.
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)

def polygon_coords_3d(geom) -> np.ndarray:
    # Extract 3-D coordinates from a Shapely polygon geometry. Expects POLYGON Z. Returns Nx3 array.
    coords = np.asarray(geom.exterior.coords, dtype=float)
    if coords.shape[1] != 3:
        raise ValueError("Expected POLYGON Z geometry")
    return coords

@njit(cache=True)
def _newell_normal(pts) -> np.ndarray:
    """
    Newell's method: returns unnormalised normal
    (Numba JIT for speed since this is called for every wall polygon vertex across all tiles/images)
    """
    n = np.zeros(3)
    nv = len(pts)
    for i in range(nv):
        p0 = pts[i]
        p1 = pts[(i + 1) % nv]
        n[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
        n[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
        n[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])
    return n

def polygon_normal_newell(coords: np.ndarray) -> np.ndarray:
    # Compute polygon normal using Newell's method. Returns unit normal vector.
    pts = coords[:-1] if np.allclose(coords[0], coords[-1]) else coords
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    n   = _newell_normal(pts)
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("Degenerate polygon")
    return n / norm

def sun_vector_enu(cam: pd.Series, lat: float, lon: float,
                   _sun_vec_cache: dict | None = None) -> np.ndarray:
    """
    Return unit ENU sun vector.  Pass _sun_vec_cache={} from the caller to
    memorise across tiles of the same image (same cam.datetime, same lat/lon).
    """
    key = (cam.datetime, round(lat, 4), round(lon, 4))
    if _sun_vec_cache is not None and key in _sun_vec_cache:
        return _sun_vec_cache[key]

    dt_utc       = parse_utc_datetime(cam.datetime)
    altitude_deg = get_altitude(lat, lon, dt_utc)
    azimuth_deg  = get_azimuth(lat, lon, dt_utc)

    alt = np.deg2rad(altitude_deg)
    az  = np.deg2rad(azimuth_deg)

    s = np.array([
        np.sin(az) * np.cos(alt),
        np.cos(az) * np.cos(alt),
        np.sin(alt),
    ], dtype=float)

    s /= np.linalg.norm(s)

    if _sun_vec_cache is not None:
        _sun_vec_cache[key] = s

    return s

def wall_shadow_score(coords: np.ndarray, sun_vec: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute wall normal and illumination score (dot product with sun vector)."""
    n     = polygon_normal_newell(coords)
    n_xy  = np.array([n[0], n[1], 0.0], dtype=float)
    norm_xy = np.linalg.norm(n_xy)
    if norm_xy == 0:
        raise ValueError("Degenerate wall normal")
    n_out = n_xy / norm_xy
    illum = float(np.dot(n_out, sun_vec))
    return n_out, illum


# ------------- Depth map -------------
def render_polygon_depth(depth_map, img_coords, depth_value) -> None:
    """Rasterise the projected polygon and update the depth map with the minimum depth value at each pixel."""
    rr, cc = polygon_to_pixel_indices(img_coords, depth_map.shape[0], depth_map.shape[1])
    depth_map[rr, cc] = np.minimum(depth_map[rr, cc], depth_value)


def build_surface_depth_map(gdf, cam):
    """
    Build a depth map of the visible surfaces from the camera's perspective, 
    by projecting all polygons and keeping the minimum depth at each pixel.
    """
    height = int(cam.sensor_rows)
    width  = int(cam.sensor_cols)

    depth_map = np.full((height, width), np.inf, dtype=np.float32)

    for geom in gdf.geometry.values:
        coords = np.asarray(geom.exterior.coords, dtype=np.float64)
        img, depth = project_points_with_depth(coords, cam)

        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            continue

        depth_value = float(np.mean(depth[valid]))
        render_polygon_depth(depth_map, img, depth_value)

    return depth_map


# ------------- Project shadow walls with visibility mask------------- 
def project_shadow_walls_visible_mask(gdf, cam, closestructure, minobjsize,
                                      depth_tol=5.0, return_info=False,
                                      _sun_vec_cache: dict | None = None):
    """
    Project only shadow-side wall pixels visible to the camera.
    """
    height = int(cam.sensor_rows)
    width  = int(cam.sensor_cols)

    mask = np.zeros((height, width), dtype=bool)

    lat = float(gdf.iloc[0]["point_lat"])
    lon = float(gdf.iloc[0]["point_lon"])

    sun_vec           = sun_vector_enu(cam, lat, lon, _sun_vec_cache)
    surface_depth_map = build_surface_depth_map(gdf, cam)

    rows     = []
    surf_gdf = gdf[gdf["type"] == "Wall"] # For now, we are only looking at the walls

    # Loop over wall polygons
    for row in surf_gdf.itertuples(index=True):
        coords = polygon_coords_3d(row.geometry)
        # Check if wall faces towards the sun (shadow) or away (sunlit) via the illumination score
        _, illum = wall_shadow_score(coords, sun_vec)
        shadowed = illum <= 1e-9
        
        info = {
            "index": row.Index,
            "building_id": getattr(row, "building_id", None),
            "illumination_score": illum,
            "shadowed": shadowed,
        }
        # If not shadowed, or if shadowed but no pixels visible to camera, we skip the expensive depth comparison and just record 0 visible pixels.
        if not shadowed:
            info["visible_pixels"] = 0
            rows.append(info)
            continue
        # Project to image and get depth along camera ray
        img, depth = project_points_with_depth(coords, cam)
        valid = np.isfinite(depth) & (depth > 0)
        # If no valid pixels, skip depth comparison and record 0 visible pixels
        if not np.any(valid):
            info["visible_pixels"] = 0
            rows.append(info)
            continue
        # Check if pixels are visible (not occluded by other surfaces) by comparing depth to surface depth map. Keep those that are within depth_tol metres of the surface depth.
        poly_depth = float(np.mean(depth[valid]))
        rr, cc     = polygon_to_pixel_indices(img, height, width)
        # We check if the polygon depth is within a tolerance of the surface depth map at the projected pixels.
        visible = np.abs(surface_depth_map[rr, cc] - poly_depth) <= depth_tol
        mask[rr[visible], cc[visible]] = True

        info["visible_pixels"] = int(np.count_nonzero(visible))
        rows.append(info)

    if return_info:
        return mask, pd.DataFrame(rows), surface_depth_map, sun_vec

    mask = binary_closing(mask, structure=closestructure)
    mask = remove_small_objects(mask, max_size=minobjsize)

    return mask

#%% ------------- Project roof mask -------------
def project_roof_mask(gdf, cam):
    height = cam.sensor_rows
    width  = cam.sensor_cols
    mask   = np.zeros((height, width), dtype=bool)

    roof_gdf = gdf[gdf["type"] == "Roof"]

    for row in roof_gdf.itertuples():
        coords = polygon_coords_3d(row.geometry)
        img, depth = project_points_with_depth(coords, cam)

        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            continue

        rr, cc = polygon_to_pixel_indices(img, height, width)
        mask[rr, cc] = True

    return mask

#%% ------------- Project Building Statistics Mask -------------

def project_points(coords, cam):
    X = coords[:, 0]
    Y = coords[:, 1]
    Z = coords[:, 2]

    dX = X - cam.Xc
    dY = Y - cam.Yc
    dZ = Z - cam.Zc

    f  = cam.f_mm / cam.pixel_size
    x0 = cam.sensor_cols * 0.5 + cam.ppo_x / cam.pixel_size
    y0 = cam.sensor_rows * 0.5 + cam.ppo_y / cam.pixel_size

    n  = cam.m31 * dX + cam.m32 * dY + cam.m33 * dZ
    xa = x0 - f * (cam.m11 * dX + cam.m12 * dY + cam.m13 * dZ) / n
    ya = y0 - f * (cam.m21 * dX + cam.m22 * dY + cam.m23 * dZ) / n

    x = xa
    y = -(ya - cam.sensor_rows)

    return np.column_stack([x, y])

def project_building_stats_masks(gdf_cam, cam, closure_b, minobj):
    """
    Yields (bid, sunlit_mask) one building at a time.
    """
    height = int(cam.sensor_rows)
    width  = int(cam.sensor_cols)

    # shared masks: built once
    shadow_mask = project_shadow_walls_visible_mask(gdf_cam, cam, closure_b, minobj)
    roof_mask   = project_roof_mask(gdf_cam, cam)

    # label map: single int32 raster for all buildings
    building_ids = gdf_cam["building_id"].values
    unique_bids  = np.unique(building_ids)
    bid_to_label = {bid: i + 1 for i, bid in enumerate(unique_bids)}

    label_map = np.zeros((height, width), dtype=np.int32)
    for geom, bid in zip(gdf_cam.geometry.values, building_ids):
        coords = np.asarray(geom.exterior.coords, dtype=np.float64)
        img    = project_points(coords, cam)
        rr, cc = raster_polygon(img[:, 1], img[:, 0], (height, width))
        label_map[rr, cc] = bid_to_label[bid]

    # yield one building at a time
    for bid in unique_bids:
        building_mask = label_map == bid_to_label[bid]
        sunlit = building_mask & ~shadow_mask & ~roof_mask
        del building_mask
        yield bid, sunlit
        del sunlit