#%% Import modules
from __future__ import annotations

import numpy as np
import pandas as pd

import os
from pathlib import Path
from urllib.parse import urlsplit

import geopandas as gpd
import requests

import subprocess
from datetime import datetime
from shapely.geometry import shape
import tqdm

import math

from PIL import Image


#%% Initialise API parameters
DEFAULT_COLLECTION = "skraafotos2023"
DEFAULT_TOKEN = None 

DEFAULT_EXIFTOOL_PATH = None

#%% --------- Fetch from Skraafotos ---------
def build_item_url(collection, image_id, token):
    # Build the API URL for fetching item metadata based on collection, image ID, and token.
    return (
        f"https://api.dataforsyningen.dk/skraafoto_api/v1.0/"
        f"collections/{collection}/items/{image_id}?token={token}"
    )

def fetch_item(url):
    # Fetch item metadata from the API using the provided URL. Raises an error if the request fails.
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def download_image(item: dict, out_dir: Path, timeout_s: int = 300) -> Path:
    # Download the image associated with the given item metadata and save it to the specified output directory.
    out_dir.mkdir(parents=True, exist_ok=True)

    image_href = item.get("assets", {}).get("data", {}).get("href")
    if not image_href:
        image_href = item.get("properties", {}).get("asset:data")
    if not image_href:
        raise ValueError("No image URL found in item assets.data.href or properties['asset:data']")

    file_name = Path(urlsplit(image_href).path).name
    out_file = out_dir / file_name

    with requests.get(image_href, stream=True, timeout=timeout_s) as response:
        response.raise_for_status()
        with out_file.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)

    return out_file

def build_camera(item):
    # Build a camera metadata dictionary from the API item metadata. Extracts relevant properties and returns them as a pandas Series.
    props = item["properties"]

    interior = props["pers:interior_orientation"]
    center = props["pers:perspective_center"]
    rot = props["pers:rotation_matrix"]

    return pd.Series({

        "datetime": props["datetime"],

        "direction": props["direction"],

        "Xc": float(center[0]),
        "Yc": float(center[1]),
        "Zc": float(center[2]),

        "f_mm": float(interior["focal_length"]),
        "ppo_x": float(interior["principal_point_offset"][0]),
        "ppo_y": float(interior["principal_point_offset"][1]),

        "pixel_size": float(interior["pixel_spacing"][0]),

        "sensor_cols": int(interior["sensor_array_dimensions"][0]),
        "sensor_rows": int(interior["sensor_array_dimensions"][1]),

        "m11": float(rot[0]),
        "m12": float(rot[1]),
        "m13": float(rot[2]),

        "m21": float(rot[3]),
        "m22": float(rot[4]),
        "m23": float(rot[5]),

        "m31": float(rot[6]),
        "m32": float(rot[7]),
        "m33": float(rot[8]),
    })

#%% --------- Fetching data for tiles ---------
def get(url):
    # Get 
    response = requests.get(url)
    j = response.json()

    df = pd.json_normalize(j, 'features')
    df['geometry'] = [shape(v['geometry']) for v in j['features']]

    next_link = [v for v in j['links'] if v['rel'] == 'next']
    href = next_link[0]['href'] if next_link else None

    return df, href

def getAll(url):
    all_df = []
    next_url = url

    while True:
        print("Fetching:", next_url)
        df, next_url = get(next_url)
        all_df.append(df)

        if not next_url:
            break

    return gpd.GeoDataFrame(
        pd.concat(all_df, ignore_index=True),
        geometry='geometry',
        crs='EPSG:4326'
    )


def get_locations(bounds, collection, token):
    # Get locations within the specified bounds.
    minx, miny, maxx, maxy = bounds
    bbox = f'{minx},{miny},{maxx},{maxy}'
    crs = 'http://www.opengis.net/def/crs/EPSG/0/25832'

    url = (
        f'https://api.dataforsyningen.dk/skraafoto_api/v1.0/'
        f'collections/{collection}/items'
        f'?limit=1000&bbox={bbox}&bbox-crs={crs}&token={token}'
    )

    gdf = getAll(url)

    keep = {
        'id':'id',
        'collection':'collection',
        'properties.datetime':'date',
        'assets.thumbnail.href':'thumbnail',
        'assets.data.href':'url',
        'properties.pers:rotation_matrix':'rotation_matrix',
        'properties.pers:perspective_center':'perspective_center',
        'properties.pers:interior_orientation.camera_id':'camera_id',
        'properties.pers:interior_orientation.focal_length':'focal_length',
        'properties.pers:interior_orientation.pixel_spacing':'pixel_spacing',
        'properties.pers:interior_orientation.calibration_date':'calibration_date',
        'properties.pers:interior_orientation.principal_point_offset':'principal_point_offset',
        'properties.pers:interior_orientation.sensor_array_dimensions':'sensor_array_dimensions',
        'properties.pers:omega':'omega',
        'properties.pers:phi':'phi',
        'properties.pers:kappa':'kappa',
        'properties.estimated_accuracy':'estimated_accuracy'
    }

    gdf2 = gdf[keep.keys()].rename(columns=keep)
    gdf2['c_x'] = [v[0] for v in gdf2['perspective_center']]
    gdf2['c_y'] = [v[1] for v in gdf2['perspective_center']]
    gdf2['c_z'] = [v[2] for v in gdf2['perspective_center']]

    gdf2['pixel_spacing_x'] = [v[0] for v in gdf2['pixel_spacing']]
    gdf2['pixel_spacing_y'] = [v[1] for v in gdf2['pixel_spacing']]

    gdf2['sensor_array_dimensions_x'] = [v[0] for v in gdf2['sensor_array_dimensions']]
    gdf2['sensor_array_dimensions_y'] = [v[1] for v in gdf2['sensor_array_dimensions']]

    gdf2['principal_point_offset_x'] = [v[0] for v in gdf2['principal_point_offset']]
    gdf2['principal_point_offset_y'] = [v[1] for v in gdf2['principal_point_offset']]

    # Expand rotation matrix to columns
    expanded_df = gdf2['rotation_matrix'].apply(pd.Series)
    expanded_df.columns = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
    gdf2 = pd.concat([gdf2.drop('rotation_matrix', axis=1), expanded_df], axis=1)


    gdf2 = gdf2.drop(columns=['perspective_center','pixel_spacing','sensor_array_dimensions','principal_point_offset'])

    return gdf2

#%% --------- Process images for iTwin ---------
def compute_itwin_opk(row):
    """
    Make .csv with camera pose parameters in iTwin's convention
    """

    # Original world-camera matrix from API
    R = np.array([
        [row["r11"], row["r12"], row["r13"]],
        [row["r21"], row["r22"], row["r23"]],
        [row["r31"], row["r32"], row["r33"]],
    ], dtype=float)

    # Bentley camera-orientation conversion for "X right, Y up"
    O = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ], dtype=float)

    # 90deg CW rotation about camera Z for pixel rotation
    alpha = -math.pi / 2
    c, s = math.cos(alpha), math.sin(alpha)
    Rz_cw = np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    # If portrait, we rotated the pixels CW, so rotate the camera frame accordingly
    use_flip = row["sensor_array_dimensions_x"] < row["sensor_array_dimensions_y"]
    R_pre = (Rz_cw @ R) if use_flip else R

    # Convert into iTwin's expected rotation convention
    R_itwin = O @ R_pre

    # Extract OPK
    r31 = float(R_itwin[2, 0])
    r31 = max(-1.0, min(1.0, r31))

    phi = math.asin(r31)
    omega = math.atan2(-float(R_itwin[2, 1]), float(R_itwin[2, 2]))
    kappa = math.atan2(-float(R_itwin[1, 0]), float(R_itwin[0, 0]))

    return {
        "omega_itwin": math.degrees(omega),
        "phi_itwin":   math.degrees(phi),
        "kappa_itwin": math.degrees(kappa),

        # optional: store the matrix too (useful for debugging)
        "r11_itwin": R_itwin[0, 0], "r12_itwin": R_itwin[0, 1], "r13_itwin": R_itwin[0, 2],
        "r21_itwin": R_itwin[1, 0], "r22_itwin": R_itwin[1, 1], "r23_itwin": R_itwin[1, 2],
        "r31_itwin": R_itwin[2, 0], "r32_itwin": R_itwin[2, 1], "r33_itwin": R_itwin[2, 2],
    }

def rotate_if_portrait(path):
    # Rotate the image 90 degrees counterclockwise if it's in portrait orientation, preserving EXIF data if present.
    with Image.open(path) as img:
        if img.width < img.height:
            exif = img.info.get("exif")
            img = img.rotate(-90, expand=True)

            if exif:
                img.save(path, exif=exif)
            else:
                img.save(path)

def build_exif_command(row, image_path, EXIFTOOL_PATH=DEFAULT_EXIFTOOL_PATH):
    # Build the command to update EXIF and XMP metadata for the given image based on the camera parameters in the row.
    portrait = row.sensor_array_dimensions_x < row.sensor_array_dimensions_y

    W = float(row.sensor_array_dimensions_y)
    H = float(row.sensor_array_dimensions_x)

    if portrait:
        W, H = H, W

    pitch_x = float(row.pixel_spacing_x)
    pitch_y = float(row.pixel_spacing_y)

    if portrait:
        pitch_x, pitch_y = pitch_y, pitch_x

    fx = float(row.focal_length) / pitch_x
    fy = float(row.focal_length) / pitch_y

    dx = float(row.principal_point_offset_x)
    dy = float(row.principal_point_offset_y)

    if portrait:
        dx, dy = dy, dx

    cx_off = dx / pitch_x
    cy_off = dy / pitch_y

    cx = (W / 2.0) + cx_off
    cy = (H / 2.0) + cy_off

    cal_date = str(row.calibration_date)

    dewarp = (
        f"{cal_date};"
        f"{fx:.5f},{fy:.5f},"
        f"{cx_off:.5f},{cy_off:.5f},"
        f"0,0,0,0,0"
    )

    ts = datetime.fromisoformat(row.date.replace("Z", "+00:00")) \
             .strftime("%Y:%m:%d %H:%M:%S")

    px_pr_cm = 10 * (1 / row.pixel_spacing_x)

    cmd = [
        "perl",
        EXIFTOOL_PATH,
        "-overwrite_original",

        # EXIF
        f"-EXIF:DateTimeOriginal={ts}",
        f"-EXIF:Make=Camera",
        f"-EXIF:Model={row.camera_id}",
        f"-EXIF:FocalLength={row.focal_length:.5f}",
        "-EXIF:FocalPlaneResolutionUnit#=3",
        f"-EXIF:FocalPlaneXResolution={px_pr_cm:.8f}",
        f"-EXIF:FocalPlaneYResolution={px_pr_cm:.8f}",

        # DJI XMP
        f"-XMP-drone-dji:CalibratedFocalLength={fx:.5f}",
        f"-XMP-drone-dji:CalibratedOpticalCenterX={cx:.5f}",
        f"-XMP-drone-dji:CalibratedOpticalCenterY={cy:.5f}",
        "-XMP-drone-dji:DewarpFlag=0",
        f"-XMP-drone-dji:DewarpData={dewarp}",

        image_path,
    ]

    return cmd, portrait


def run_cmd(cmd):
    print("RUNNING:", " ".join(cmd))
    return subprocess.call(cmd)

def make_csv(df, outdir):
    df['pixel_spacing'] = df.pixel_spacing_x
    df['px_pr_cm'] = 10*(1/df.pixel_spacing)

    df['sensor_size_x'] = df.sensor_array_dimensions_x * df.pixel_spacing
    df['sensor_size_y'] = df.sensor_array_dimensions_y * df.pixel_spacing


    df = df.join(df.apply(compute_itwin_opk, axis=1, result_type="expand"))


    df2 = df[["id", "c_x", "c_y", "c_z","omega_itwin","phi_itwin",'kappa_itwin']]
    df2['acc_x'] = 0.2
    df2['acc_y'] = 0.2
    df2['acc_z'] = 0.2
    df2['acc_omega'] = 5
    df2['acc_phi'] = 5
    df2['acc_kappa'] = 5

    df2.to_csv(f'{outdir}/constraints_flip_itwin.csv',sep=',',index=False,header=False)

#%% --------- Main function for iTwin compatible post-processing ---------
def gdfstamp(df, outdir):

    make_csv(df, outdir)

    print(f"\nFound {len(df)} camera records\n")

    processed = 0

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        img_path = Path(outdir) / f"{row.id}.jpg"

        if not img_path.is_file():
            continue

        cmd, portrait = build_exif_command(row, str(img_path))

        if portrait:
            rotate_if_portrait(img_path)

        run_cmd(cmd)
        processed += 1

    print(f"\nDone. Updated {processed} images.\n")