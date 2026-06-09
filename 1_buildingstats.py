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

import os
from pathlib import Path
import sys

from fetch import *
from gdfproj import *
from reinhard import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("stats.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

#%% Profiling helpers (lightweight, stdlib-only)
def peak_rss_mb() -> float:
    """Peak resident set size of the *current* process in MB.
    On Linux ru_maxrss is in kB; on macOS it is in bytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024.0 if platform.system() == "Linux" else rss / (1024.0 ** 2)

# Ordered stage names for this script's per-image timing breakdown.
STAGE_NAMES = ["cull", "load_img", "masks", "building_loop"]

#%% Take area arguments
if len(sys.argv) < 2:
    print("Usage: python stats.py <area>")
    sys.exit(1)

area = sys.argv[1]
log.info(f"=== stats.py starting for area: {area} ===")

#%% Define environment
image_dir = Path(f"images/{area}")
cam_dir   = Path(f"cams/{area}")
stats_dir = Path(f"stats/{area}")
stats_dir.mkdir(parents=True, exist_ok=True)

#%% Define parameters
closure_b = np.ones((6, 6)) # small closure kernel to exclude self-shadow pixels close to the building footprint boundary
minobj    = 5000 # minimum object size in pixels for the closure operation for the mask. Excludes small shadow areas from cars or other areas not of interest
minbuilding_pixels = 2500 # minimum number of pixels required for a building to be included in the stats. Excludes buildings that have too few pixels for reliable stats. 
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 7)) # tune based on system and workload. More workers can speed up processing but also increase memory usage, so adjust based on available resources.

#%% Load cameras from disk
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

image_ids = list(cams.keys())
north_ids = [iid for iid in image_ids if cams[iid]["direction"] == "north"]
log.info(f"Cameras loaded: {len(image_ids)} total, {len(north_ids)} north-facing")

#%% Load GDF once in the main process
log.info("Loading GDF ...")
gdf               = gpd.read_file(f"poly/{area}.gpkg")
z_ground          = get_ground_z(gdf)
n_buildings_total = gdf["building_id"].nunique()
log.info(f"GDF loaded: {len(gdf)} polygons, "
         f"{n_buildings_total} unique buildings, z_ground={z_ground:.1f}")

#%% Per-image worker
# Self-shadowed walls (shadow_walls) and roof pixels (roof_mask) are computed once per image and held in memory for the loop over buildings, to avoid redundant computation. 
def process_image(image_id: str,
                  cam: pd.Series,
                  gdf: gpd.GeoDataFrame,
                  z_ground: float,
                  image_dir: Path,
                  closure_b: np.ndarray,
                  minobj: int,
                  ) -> tuple[str, dict, dict]:
    """
    Process one north image. Returns (image_id, {bid: (mean, std)}, prof).
    prof carries per-stage timings (s), counts, and peak RSS for scalability analysis.
    Runs in a worker process — no shared state.
    """
    result = {}
    t = {s: 0.0 for s in STAGE_NAMES}   # per-stage seconds
    t_img0 = time.perf_counter()

    # 1. spatial cull (crop the GDF to the camera footprint + margin, to reduce the number of buildings we loop over in the next stage)
    _t = time.perf_counter()
    gdf_cam = cull_gdf_to_camera(gdf, cam, z_ground=z_ground, margin_m=10.0)
    t["cull"] += time.perf_counter() - _t
    n_buildings_in_view = int(gdf_cam["building_id"].nunique()) if len(gdf_cam) else 0
    if len(gdf_cam) == 0:
        prof = {"image_id": image_id, "total_s": time.perf_counter() - t_img0,
                "n_buildings_view": 0, "n_buildings_stored": 0,
                "peak_rss_mb": peak_rss_mb(), **t}
        return image_id, result, prof

    # 2. load image
    _t = time.perf_counter()
    img_path = image_dir / f"{image_id}.jpg"
    img_lab  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2LAB) if img_path.exists() else None
    t["load_img"] += time.perf_counter() - _t
    if img_lab is None:
        prof = {"image_id": image_id, "total_s": time.perf_counter() - t_img0,
                "n_buildings_view": n_buildings_in_view, "n_buildings_stored": 0,
                "peak_rss_mb": peak_rss_mb(), **t}
        return image_id, result, prof

    # 3. match image dimensions to camera metadata
    cam_h = int(cam.sensor_rows)
    cam_w = int(cam.sensor_cols)
    if img_lab.shape[0] == cam_w and img_lab.shape[1] == cam_h:
        img_lab = cv2.rotate(img_lab, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 4. projected shadow and roof masks. Built once, held for loop
    _t = time.perf_counter()
    shadow_walls = project_shadow_walls_visible_mask(gdf_cam, cam, closure_b, minobj)
    roof_mask    = project_roof_mask(gdf_cam, cam)
    t["masks"] += time.perf_counter() - _t

    # 5. iterate over buildings
    _t = time.perf_counter()
    for bid, sunlit in project_building_stats_masks(gdf_cam, cam, closure_b, minobj):
        pixels = img_lab[sunlit]
        del sunlit

        if pixels.shape[0] < minbuilding_pixels:
            del pixels
            continue
        
        # Get mean and std per LAB channel for sunlit pixels of this building and store in result dict
        result[bid] = (
            pixels.mean(axis=0).astype(np.float32),
            (pixels.std(axis=0) + 1e-6).astype(np.float32),
        )
        del pixels
    t["building_loop"] += time.perf_counter() - _t

    del img_lab, shadow_walls, roof_mask, gdf_cam
    gc.collect()

    prof = {"image_id": image_id, "total_s": time.perf_counter() - t_img0,
            "n_buildings_view": n_buildings_in_view, "n_buildings_stored": len(result),
            "peak_rss_mb": peak_rss_mb(), **t}
    return image_id, result, prof

#%% Parallel execution
log.info(f"Processing {len(north_ids)} north images with {N_WORKERS} workers ...")

all_results: dict = {}   # bid -> (mean, std). First image per building wins
completed = 0

# --- profiling state ---
profile_rows: list[dict] = []
run_t0 = time.perf_counter()

with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    # Submit all tasks to the pool and keep track of which future corresponds to which image_id for logging purposes. The futures will complete in arbitrary order, so we use as_completed to process them as they finish.
    futures = {
        pool.submit(
            process_image,
            image_id, cams[image_id], gdf, z_ground,
            image_dir, closure_b, minobj,
        ): image_id
        for image_id in north_ids
    }
    # As each future completes, we retrieve the result and update our all_results dict. We also log progress and any errors that occur during processing.
    for future in as_completed(futures):
        image_id = futures[future]
        completed += 1
        try:
            _, result, prof = future.result()
            profile_rows.append(prof)
            new_bids = 0
            for bid, stats in result.items():
                if bid not in all_results:
                    all_results[bid] = stats
                    new_bids += 1
            log.info(f"  [{completed}/{len(north_ids)}] {image_id}: "
                     f"+{new_bids} new buildings "
                     f"({len(all_results)}/{n_buildings_total} total) "
                     f"[{prof['total_s']:.2f}s, {prof['peak_rss_mb']:.0f}MB]")
        except Exception as e:
            log.error(f"  [{completed}/{len(north_ids)}] {image_id}: FAILED — {e}")

wall_s = time.perf_counter() - run_t0
log.info(f"Stats complete: {len(all_results)} / {n_buildings_total} buildings")

#%% Write per-image profile CSV + summary
prof_csv = stats_dir / "profile_buildingstats.csv"
fieldnames = ["image_id", "total_s", *STAGE_NAMES,
              "n_buildings_view", "n_buildings_stored", "peak_rss_mb"]
with open(prof_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in profile_rows:
        w.writerow({k: row.get(k, "") for k in fieldnames})
log.info(f"Per-image profile written to {prof_csv}")

if profile_rows:
    totals   = [r["total_s"] for r in profile_rows]
    sum_proc = sum(totals)
    log.info("SCALABILITY SUMMARY (1_buildingstats)")
    log.info(f"  area={area}  images={len(profile_rows)}  workers={N_WORKERS}")
    log.info(f"  wall-clock        : {wall_s:.1f}s ({wall_s/60:.1f} min)")
    log.info(f"  sum per-image time: {sum_proc:.1f}s "
             f"(parallel speedup x{sum_proc/wall_s:.2f}, "
             f"efficiency {sum_proc/wall_s/N_WORKERS*100:.0f}%)")
    log.info(f"  per-image total_s : median={statistics.median(totals):.2f}  "
             f"min={min(totals):.2f}  max={max(totals):.2f}")
    log.info(f"  throughput        : {len(profile_rows)/wall_s:.2f} images/s")
    log.info("  stage share of summed per-image time:")
    for s in STAGE_NAMES:
        share = sum(r.get(s, 0.0) for r in profile_rows)
        log.info(f"    {s:<14}: {share:7.1f}s ({share/sum_proc*100:4.1f}%)")
    peak = max(r["peak_rss_mb"] for r in profile_rows)
    log.info(f"  peak worker RSS   : {peak:.0f} MB "
             f"(~{peak*N_WORKERS/1024:.1f} GB across {N_WORKERS} workers)")

#%% Save results to disk
stats_path = stats_dir / "building_stats.npz"
log.info(f"Saving to {stats_path} ...")
np.savez(
    stats_path,
    bids  = np.array(list(all_results.keys())),
    means = np.array([v[0] for v in all_results.values()]),
    stds  = np.array([v[1] for v in all_results.values()]),
)
log.info("Saved. Done.")