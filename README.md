# Shadow Removal for Oblique Aerial Imagery
Fully automated pipeline for shadow compensation in high-resolution oblique aerial imagery, developed for the MSc thesis "Shadow Removal and Mesh Improvements for Photogrammetry-based 3D Modelling of Built Environments" (Juliane Bjørn Budde, 2026).

## About
The method combines LiDAR-derived building geometry, camera metadata, and solar position to detect and compensate two kinds of shadows in oblique imagery:
- Self-shadows building walls. Detected geometrically by projecting wall polygons into each image and comparing wall normals to the sun directing, with depth-based visibility filtering. Compensated by Reinhard colour transfer in CIELAB space against sunlit statistis of the same building gathered from nort-facing images.
- Cast grond shadows. Detected with the radiometric Shadow Detection Index (SDI, Liu, X. et al. (2022)). Compensated by Reinhard colour transfer in CIELAB space against sunlit statistis of the ssurroundings. Penumbra is blended with random sampling to suppress harsh boundary transitions.

The compensated images are inteded as input to photogrammetric reconstruction in iTwin Capture (Bentley), and the pipeline therefore includes metadata stamping suited for iTwin Capture input.

## Pipeline Overview
0. 0_download.py: Query the Skråfotos API for all images in the area bounding box, download the TIFF's, convert to JPG, and stamp camera metadata (EXIF/XMP).
1. 1_buildingstats.py: Pass over north-facing images, accumulating sunlit-facade colour statistics (mean/std in CIELAB) for each building, and save to disk.
2. 2_compensate.py: For each image, build geometry-based wall-shadow and roof masks, compensate self-shadowed walls against the stored building statistics, detect cast ground shadows with SDI, and compensate them against local surroundings.
3. 3_stamp.py: Re-stamp the compensated images with the same camera metadata as the originals, so the original and compensated reconstructions differ only in pixel values, where shadow was detected.

## Modules
- fetch.py: Skråfotos STAC API access, camera-record construction, and EXIF/XMP/constraints-file stamping for iTwin Capture.
- gdfproj.py: Collinearity projection, camera ground-footprint culling, solar vector (via pysolar), Newell wall normals, depth maps, and projection of wall, roof and building masks.
- reinhard.py: Reinhard colour transfer for wall and cast shadows with penumbra-blending.
- sdi.py: Shadow Detection Index (Liu, X. et al. (2022)) and filters including vegetation removal, area filtering, building-proximity filtering.

## Setup
### Python environment
conda env create -f environment.yml
conda activate shadowcomp

### External tools (not installed by conda)
ImageMagick and ExifTool must be available as command-line tools for stage 0. and 3. in the pipeline. When installed, set the DEFAULT_EXIFTOOL_PATH in fetch.py.

### API token
Access to the Skråfotos API requires a token from Datafordeler, which is not provided here. Obtain a token from Datafordeler and set DEFAULT_TOKEN in fetch.py

### Input data
Needs LiDAR-derived building geometry, as "poly/"area".gpkg". The GeoPackage must contain 3D (Polygon Z) geometries with the following attributes:
- building_id: unique identifier shared by all polygons of one building.
- type: "Wall" and "Roof" are used in this pipeline.

The remaining data including images and camera metadata are downloaded from the Danish Skråfotos API when DEFAULT_TOKEN is specified in fetch.py.

## Usage
Run the stages in order, passing the area name matching the GeoPackage of interest (matching "poly/"area".gpkg"):
python 0_download.py "area" <br>
python 1_buildingstats.py "area" <br>
python 2_compensate.py "area" <br>
python 3_stamp.py "area" <br>

For the results in the thesis, the pipeline was run on DTU's HPC for the area "nordvest" with <br>
bsub < run_download.sh <br>
bsub < run_buildingstats.sh <br>
bsub < run_compensate.sh <br>
bsub < run_stamp.sh <br>
The results for "aarhus" were run in the same manner, by swapping out any "nordvest" in the bash scripts by "aarhus".

## Directory
poly/"area".gpkg, input LiDAR building model <br>
images/"area"/, output of stage 0 containing downloaded and coverted JPGS <br>
cams/"area", output of stage 0 containing per-image camera metdata JSON <br>
stats/"area", output of stage 1 containg sunlit CIELAB statistics in an .npz <br>
comp/"area", output of stage 2 (re-stamped in stage 3) containing shadow-compensated JPGs <br>
Failure logs and stage logs are written to the working directory, and batch-ouputs are written to batch_output.

## Parameters
Key parameters are defined near the top of each stage script and in the module defaults (see the thesis Appendix C for the full parameter overview).
