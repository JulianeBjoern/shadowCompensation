#%%
from pathlib import Path
import pandas as pd
import tqdm
import sys

#%% Arguments
if len(sys.argv) < 2:
    print("Usage: python 3_stamp.py <area>")
    sys.exit(1)

area = sys.argv[1]

from fetch import (
    DEFAULT_COLLECTION,
    DEFAULT_TOKEN,
    build_item_url,
    fetch_item,
    gdfstamp,
)

#%% Initialise API parameters
COLLECTION = DEFAULT_COLLECTION
TOKEN      = DEFAULT_TOKEN
#%% Define environment
COMP_DIR = Path(f"comp/{area}") # Folder containing compensated JPGs
OUTDIR = COMP_DIR # Output folder. Stamp in place

#%% Find image ids from compensated images
image_paths = sorted(COMP_DIR.glob("*.jpg"))
if not image_paths:
    raise RuntimeError(f"No .jpg images found in {COMP_DIR}")
image_ids = [p.stem for p in image_paths]
print(f"Found {len(image_ids)} images")

#%% Fetch metadata for each image
rows = []
for image_id in tqdm.tqdm(image_ids):
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
        print(f"Failed for {image_id}: {e}")

# Build dataframe
df = pd.DataFrame(rows)
print(f"\nSuccessfully fetched {len(df)} camera records")

# Stamp EXIF and GDF
gdfstamp(df, OUTDIR)

print("\nDone.")