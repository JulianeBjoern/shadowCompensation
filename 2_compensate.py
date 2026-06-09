#%% Load modules
from __future__ import annotations
import json
import logging
import gc
import csv
import time
import platform
import resource
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import cv2

from pathlib import Path
import sys

from fetch import *
from gdfproj import *
from reinhard import *
from sdi import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("compensate.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

#%% Profiling helpers (lightweight, stdlib-only)
def peak_rss_mb() -> float:
    """Peak resident set size of the current process in MB.
    On Linux ru_maxrss is in kB; on macOS it is in bytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024.0 if platform.system() == "Linux" else rss / (1024.0 ** 2)

# Ordered stage names for this script's per-image timing breakdown.
STAGE_NAMES = ["cull", "load_img", "masks",
               "wall_comp_loop", "sdi_cast", "cast_comp", "write"]
#%% Arguments
if len(sys.argv) < 2:
    print("Usage: python compensate.py <area>")
    sys.exit(1)

area = sys.argv[1]
log.info(f"compensate.py starting for area: {area}")

#%% Environment
image_dir = Path(f"images/{area}")
cam_dir   = Path(f"cams/{area}")
stats_dir = Path(f"stats/{area}")
out_dir   = Path(f"comp/{area}")
out_dir.mkdir(parents=True, exist_ok=True)

#%% Parameters
closure_b = np.ones((6, 6))
minobj    = 5000
N_WORKERS = 6 # N_WORKERS = os.cpu_count(), but set to 6 to avoid memory issues on HPC with 8 cores but only 64GB RAM.

#%% Load cameras
image_ids = sorted(f.stem for f in image_dir.glob("*.jpg"))
log.info(f"Found {len(image_ids)} .jpg files in {image_dir}")

cams = {}
for image_id in image_ids:
    cam_path = cam_dir / f"{image_id}.json"
    if not cam_path.exists():
        log.warning(f"  No camera file for {image_id}, skipping")
        continue
    with open(cam_path) as f:
        cams[image_id] = pd.Series(json.load(f))

image_ids = [iid for iid in image_ids if iid in cams]
log.info(f"Cameras loaded for {len(image_ids)} images")

#%% Load GDF
log.info("Loading GDF ...")
gdf      = gpd.read_file(f"poly/{area}.gpkg")
z_ground = get_ground_z(gdf)
log.info(f"GDF loaded: {len(gdf)} polygons, z_ground={z_ground:.1f}")

#%% Load building statistics
stats_path = stats_dir / "building_stats.npz"
log.info(f"Loading building stats from {stats_path} ...")
data = np.load(stats_path, allow_pickle=False)
building_stats = {
    bid: (mean, std)
    for bid, mean, std in zip(data["bids"], data["means"], data["stds"])
}
log.info(f"Loaded stats for {len(building_stats)} buildings")

#%% Helper: iterate per-building masks one at a time (label-map trick)
def _iter_building_masks(gdf_cam, cam):
    # Get (bid, building_mask) for each building in the camera view, one at a time.
    from skimage.draw import polygon as raster_polygon

    height = int(cam.sensor_rows)
    width  = int(cam.sensor_cols)

    building_ids = gdf_cam["building_id"].values
    unique_bids  = np.unique(building_ids)
    bid_to_label = {bid: i + 1 for i, bid in enumerate(unique_bids)}

    # Rasterize all buildings into a single label map, where pixel values correspond to building labels, then iterate over unique building IDs
    label_map = np.zeros((height, width), dtype=np.int32)
    for geom, bid in zip(gdf_cam.geometry.values, building_ids):
        coords = np.asarray(geom.exterior.coords, dtype=np.float64)
        img    = project_points(coords, cam)
        rr, cc = raster_polygon(img[:, 1], img[:, 0], (height, width))
        label_map[rr, cc] = bid_to_label[bid]

    for bid in unique_bids:
        yield bid, label_map == bid_to_label[bid]


#%% Per-image worker. Runs in a subprocess, no shared state
def process_image(image_id: str,
                  cam: pd.Series,
                  gdf: gpd.GeoDataFrame,
                  building_stats: dict,
                  z_ground: float,
                  image_dir: Path,
                  out_dir: Path,
                  closure_b: np.ndarray,
                  minobj: int,
                  ) -> tuple[str, bool, str, dict]:
    """Full compensation pipeline for one image.
    Returns (image_id, ok, msg, prof) where prof holds per-stage timings (s),
    counts, and peak RSS for scalability analysis."""

    t = {s: 0.0 for s in STAGE_NAMES}
    t_img0 = time.perf_counter()

    def _prof(n_view=0, n_comp=0):
        return {"image_id": image_id, "total_s": time.perf_counter() - t_img0,
                "n_buildings_view": n_view, "n_buildings_comp": n_comp,
                "peak_rss_mb": peak_rss_mb(), **t}

    out_path = out_dir / f"{image_id}.jpg"
    if out_path.exists():
        return image_id, True, "skipped (already exists)", _prof()

    try:
        # 1. spatial cull
        _t = time.perf_counter()
        gdf_cam = cull_gdf_to_camera(gdf, cam, z_ground=z_ground, margin_m=10.0)
        t["cull"] += time.perf_counter() - _t
        n_view = int(gdf_cam["building_id"].nunique()) if len(gdf_cam) else 0

        # 2. load image
        _t = time.perf_counter()
        img_path = image_dir / f"{image_id}.jpg"
        img_bgr  = cv2.imread(str(img_path))
        t["load_img"] += time.perf_counter() - _t
        if img_bgr is None:
            return image_id, False, f"could not read {img_path}", _prof(n_view)
        
        # Match image dimensions to camera metadata
        cam_h = int(cam.sensor_rows)
        cam_w = int(cam.sensor_cols)
        if img_bgr.shape[0] == cam_w and img_bgr.shape[1] == cam_h:
            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 3. Convert to LAB for compensation
        img_lab_comp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        del img_bgr
        H, W = img_lab_comp.shape[:2]

        # 4. full masks
        _t = time.perf_counter()
        shadow_walls = project_shadow_walls_visible_mask(
            gdf_cam, cam, closure_b, minobj)
        roof_mask    = project_roof_mask(gdf_cam, cam)
        t["masks"] += time.perf_counter() - _t

        # 5. wall compensation + building_union (single pass)
        _t = time.perf_counter()
        building_union = np.zeros((H, W), dtype=bool)
        n_comp = 0

        for bid, building_mask in _iter_building_masks(gdf_cam, cam):
            building_union |= building_mask

            if bid in building_stats:
                mean_sun, std_sun = building_stats[bid]
                wall_shadow_mask  = building_mask & shadow_walls
                reinhard_building(img_lab_comp, wall_shadow_mask,
                                  mean_sun, std_sun)
                del wall_shadow_mask
                n_comp += 1

            del building_mask
        t["wall_comp_loop"] += time.perf_counter() - _t

        # 6. cast shadow detection
        _t = time.perf_counter()
        img_bgr = cv2.cvtColor(img_lab_comp, cv2.COLOR_LAB2BGR)
        shadow_cast  = sdi_filtered(img_bgr, building_union)
        del img_bgr
        shadow_cast &= ~building_union
        t["sdi_cast"] += time.perf_counter() - _t

        # 7. cast shadow compensation
        _t = time.perf_counter()
        reinhard_cast(img_lab_comp, shadow_cast,
                      building_union, shadow_walls | roof_mask)
        t["cast_comp"] += time.perf_counter() - _t

        # 8. write output
        _t = time.perf_counter()
        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(img_lab_comp, cv2.COLOR_LAB2BGR),
        )
        t["write"] += time.perf_counter() - _t

        del img_lab_comp, shadow_cast, building_union, shadow_walls, roof_mask, gdf_cam
        
        gc.collect()

        return image_id, True, "ok", _prof(n_view, n_comp)

    except Exception as e:
        return image_id, False, str(e), _prof()


#%% Parallel execution
log.info(f"Processing {len(image_ids)} images with {N_WORKERS} workers ...")

completed = 0
failed    = []

# --- profiling state ---
profile_rows: list[dict] = []
run_t0 = time.perf_counter()

with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    # Submit all tasks to the pool and keep track of which future corresponds to which image_id for logging purposes. The futures will complete in arbitrary order, so we use as_completed to process them as they finish.
    futures = {
        pool.submit(
            process_image,
            image_id, cams[image_id], gdf, building_stats,
            z_ground, image_dir, out_dir, closure_b, minobj,
        ): image_id
        for image_id in image_ids
    }
    # As each future completes, we retrieve the result and update our all_results dict. We also log progress and any errors that occur during processing.
    for future in as_completed(futures):
        image_id = futures[future]
        completed += 1
        try:
            _, ok, msg, prof = future.result()
            prof["status"] = msg
            profile_rows.append(prof)
            if ok:
                log.info(f"  [{completed}/{len(image_ids)}] {image_id}: {msg} "
                         f"[{prof['total_s']:.2f}s, {prof['peak_rss_mb']:.0f}MB]")
            else:
                log.error(f"  [{completed}/{len(image_ids)}] {image_id}: FAILED — {msg}")
                failed.append(image_id)
        except Exception as e:
            log.error(f"  [{completed}/{len(image_ids)}] {image_id}: EXCEPTION — {e}")
            failed.append(image_id)

wall_s = time.perf_counter() - run_t0

#%% Write per-image profile CSV + summary
prof_csv = out_dir / "profile_compensate.csv"
fieldnames = ["image_id", "status", "total_s", *STAGE_NAMES,
              "n_buildings_view", "n_buildings_comp", "peak_rss_mb"]
with open(prof_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in profile_rows:
        w.writerow({k: row.get(k, "") for k in fieldnames})
log.info(f"Per-image profile written to {prof_csv}")

# Summarise only genuinely processed images (exclude skipped, which are ~0s).
proc = [r for r in profile_rows if not str(r.get("status", "")).startswith("skipped")]
if proc:
    totals   = [r["total_s"] for r in proc]
    sum_proc = sum(totals)
    log.info("SCALABILITY SUMMARY (2_compensate)")
    log.info(f"  area={area}  processed={len(proc)} "
             f"(skipped={len(profile_rows)-len(proc)})  workers={N_WORKERS}")
    log.info(f"  wall-clock        : {wall_s:.1f}s ({wall_s/60:.1f} min)")
    log.info(f"  sum per-image time: {sum_proc:.1f}s "
             f"(parallel speedup x{sum_proc/wall_s:.2f}, "
             f"efficiency {sum_proc/wall_s/N_WORKERS*100:.0f}%)")
    log.info(f"  per-image total_s : median={statistics.median(totals):.2f}  "
             f"min={min(totals):.2f}  max={max(totals):.2f}")
    log.info(f"  throughput        : {len(proc)/wall_s:.2f} images/s")
    log.info("  stage share of summed per-image time:")
    for s in STAGE_NAMES:
        share = sum(r.get(s, 0.0) for r in proc)
        log.info(f"    {s:<14}: {share:7.1f}s ({share/sum_proc*100:4.1f}%)")
    peak = max(r["peak_rss_mb"] for r in proc)
    log.info(f"  peak worker RSS   : {peak:.0f} MB "
             f"(~{peak*N_WORKERS/1024:.1f} GB across {N_WORKERS} workers)")

#%% Summary
log.info(f" compensate.py complete: "
         f"{len(image_ids) - len(failed)}/{len(image_ids)} succeeded ")

if failed:
    fail_path = Path(f"failed_compensate_{area}.txt")
    fail_path.write_text("\n".join(failed))
    log.warning(f"{len(failed)} failures written to {fail_path}")