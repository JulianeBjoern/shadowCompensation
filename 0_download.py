#%% Load modules
from __future__ import annotations
import json
import logging
import subprocess

from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys

from fetch import (
    DEFAULT_COLLECTION,
    DEFAULT_TOKEN,
    build_item_url,
    fetch_item,
    get_locations,
    download_image,
    build_camera,
    gdfstamp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("download.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

#%% Take area arguments

if len(sys.argv) < 2:
    print("Usage: python 0_download_stamp.py <area>")
    sys.exit(1)

area = sys.argv[1]
gdf  = gpd.read_file(f"poly/{area}.gpkg")

#%% Define environment
# temporary TIF landing zone
tif_dir = Path(f"images/{area}/_tif")
tif_dir.mkdir(parents=True, exist_ok=True)
# final JPGs + EXIF
jpg_dir = Path(f"images/{area}")
jpg_dir.mkdir(parents=True, exist_ok=True)
# camera metadata for use in further processing (so we only fetch each item once)
cam_dir = Path(f"cams/{area}")
cam_dir.mkdir(parents=True, exist_ok=True)

#%% Initialise API parameters
TOKEN      = DEFAULT_TOKEN
COLLECTION = DEFAULT_COLLECTION
BUFFER     = 500   # e.g. 500 m each side = 1 km x 1 km box
if area == "aarhus":
    BUFFER = 1000 # Aarhus is more spread out, so use a larger buffer to get more images.

#%% Bounding box: centroid of the gpkg
total_bounds = gdf.total_bounds  # (minx, miny, maxx, maxy) in GDF CRS
cx = (total_bounds[0] + total_bounds[2]) / 2
cy = (total_bounds[1] + total_bounds[3]) / 2
bbox_geom = (cx - BUFFER, cy - BUFFER, cx + BUFFER, cy + BUFFER)  # EPSG:25832
log.info(f"Area centroid: ({cx:.1f}, {cy:.1f})  bbox: {bbox_geom}")
#%% Query API for images in the bounding box
gdf_api   = get_locations(bbox_geom, COLLECTION, TOKEN)
image_ids = gdf_api["id"].tolist()
log.info(f"API returned {len(image_ids)} images")

#%% Download worker
def download_one(image_id: str) -> tuple[str, bool, str]:
    """
    Download TIF + camera given one image.
    Returns (image_id, success, message).
    Skips if the final stamped JPG already exists.
    """

    # Check if JPG already exists and is non-empty
    jpg_path = jpg_dir / f"{image_id}.jpg"
    if jpg_path.exists() and jpg_path.stat().st_size > 0:
        return image_id, True, "skipped (already exists)"
    # Check if camera JSON already exists and is non-empty
    cam_path = cam_dir / f"{image_id}.json"

    # We still want to fetch the camera if the JSON is missing, even if the JPG exists, because we need the camera for stamping.
    try:
        item = fetch_item(build_item_url("skraafotos" + image_id[:4], image_id, TOKEN))
        cam  = build_camera(item)
        # Save camera
        with open(cam_path, "w") as f:
            json.dump(cam.to_dict(), f, indent=2)
        # Download TIF into temporary folder
        download_image(item, tif_dir)

        return image_id, True, "ok"

    except Exception as e:
        return image_id, False, str(e)

#%% Parallel downloads
MAX_WORKERS = 4
failed_download = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {pool.submit(download_one, iid): iid for iid in image_ids}
    for future in as_completed(futures):
        iid, ok, msg = future.result()
        if ok:
            log.info(f"  {iid}: {msg}")
        else:
            log.error(f"  {iid}: FAILED — {msg}")
            failed_download.append(iid)

log.info(f"Downloads complete. {len(image_ids) - len(failed_download)}/{len(image_ids)} succeeded.")

#%% Convert TIFs -> JPGs with ImageMagick, and verify output
tif_paths = sorted(tif_dir.glob("*.tif"))
log.info(f"\nConverting {len(tif_paths)} TIFFs to JPGs ...")

failed_convert = []
# Loop through TIFs and convert to JPGs, verifying each step.
for tif_path in tif_paths:
    jpg_path = jpg_dir / f"{tif_path.stem}.jpg"

    # Skip if JPG already exists and is non-empty (e.g. from a previous run)
    if jpg_path.exists() and jpg_path.stat().st_size > 0:
        tif_path.unlink()
        continue

    try:
        # Convert using ImageMagick with some basic enhancements (auto-orient, contrast, saturation).
        cmd = [
            "convert",
            f"{tif_path}[0]",
            "-auto-orient",
            "-colorspace", "sRGB",
            "-contrast-stretch", "0.5%x0.5%",
            "-modulate", "100,140,100",
            "-strip",
            "-quality", "95",
            str(jpg_path),
        ]
        # Run the conversion command
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check if the conversion succeeded
        if result.returncode != 0:
            log.error(f"  Failed conversion: {tif_path.name}\n{result.stderr}")
            if jpg_path.exists():
                jpg_path.unlink()
            failed_convert.append(tif_path.stem)
            continue

        # Verify the output
        verify = subprocess.run(
            ["identify", str(jpg_path)], capture_output=True, text=True
        )
        if verify.returncode != 0:
            log.error(f"  Invalid JPG after conversion: {jpg_path.name}")
            jpg_path.unlink()
            failed_convert.append(tif_path.stem)
            continue

        # Remove TIF only after a verified JPG exists
        tif_path.unlink()
        log.info(f"  Converted: {tif_path.name}")

    except Exception as e:
        log.error(f"  Exception during conversion: {tif_path.name}: {e}")
        if jpg_path.exists():
            jpg_path.unlink()
        failed_convert.append(tif_path.stem)

log.info(f"Conversion complete. {len(failed_convert)} failures.")

#%% Fetch metadata and stamp EXIF
jpg_paths     = sorted(jpg_dir.glob("*.jpg"))
stamp_ids     = [p.stem for p in jpg_paths]
failed_meta   = []
rows          = []

log.info(f"\nFetching metadata for {len(stamp_ids)} images ...")

for image_id in stamp_ids:
    try:
        url  = build_item_url(COLLECTION, image_id, TOKEN)
        item = fetch_item(url)
        props = item["properties"]

        row = {
            "id":         image_id,
            "collection": item["collection"],
            "date":       props["datetime"],

            "camera_id":         props["pers:interior_orientation"]["camera_id"],
            "focal_length":      props["pers:interior_orientation"]["focal_length"],
            "pixel_spacing":     props["pers:interior_orientation"]["pixel_spacing"],
            "calibration_date":  props["pers:interior_orientation"]["calibration_date"],
            "principal_point_offset":  props["pers:interior_orientation"]["principal_point_offset"],
            "sensor_array_dimensions": props["pers:interior_orientation"]["sensor_array_dimensions"],

            "perspective_center": props["pers:perspective_center"],
            "rotation_matrix":    props["pers:rotation_matrix"],
        }

        row["c_x"] = row["perspective_center"][0]
        row["c_y"] = row["perspective_center"][1]
        row["c_z"] = row["perspective_center"][2]

        row["pixel_spacing_x"] = row["pixel_spacing"][0]
        row["pixel_spacing_y"] = row["pixel_spacing"][1]

        row["sensor_array_dimensions_x"] = row["sensor_array_dimensions"][0]
        row["sensor_array_dimensions_y"] = row["sensor_array_dimensions"][1]

        row["principal_point_offset_x"] = row["principal_point_offset"][0]
        row["principal_point_offset_y"] = row["principal_point_offset"][1]

        R = row["rotation_matrix"]
        row["r11"], row["r12"], row["r13"] = R[0], R[1], R[2]
        row["r21"], row["r22"], row["r23"] = R[3], R[4], R[5]
        row["r31"], row["r32"], row["r33"] = R[6], R[7], R[8]

        rows.append(row)

    except Exception as e:
        log.error(f"  Failed metadata fetch for {image_id}: {e}")
        failed_meta.append(image_id)

df = pd.DataFrame(rows)
log.info(f"Successfully fetched {len(df)} camera records")

gdfstamp(df, jpg_dir)

#%% Summary
log.info("\n=== 0_download_stamp.py complete ===")
log.info(f"  Download failures:   {len(failed_download)}")
log.info(f"  Conversion failures: {len(failed_convert)}")
log.info(f"  Metadata failures:   {len(failed_meta)}")

all_failed = list(set(failed_download + failed_convert + failed_meta))
if all_failed:
    fail_path = Path(f"failed_{area}.txt")
    fail_path.write_text("\n".join(all_failed))
    log.warning(f"{len(all_failed)} total failures written to {fail_path}")