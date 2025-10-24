import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import geopandas as gpd
from shapely.geometry import Point

# --- Load Morocco shapefile once ---
shapefile_path = "data/maroc shape file/Morocco_ADM0_simplified.simplified(1).geojson"
gdf = gpd.read_file(shapefile_path)
morocco_geom = gdf.geometry.unary_union  # merge all polygons

# --- Function to extract binary FLS mask from NetCDF ---
def extract_FLS_maroc(nc_file):
    nc = Dataset(nc_file, 'r')
    channel_4 = nc.variables['channel_4'][:]
    channel_9 = nc.variables['channel_9'][:]
    channel_10 = nc.variables['channel_10'][:]
    latitudes = nc.variables['lat'][:]
    longitudes = nc.variables['lon'][:]
    date_time = nc.getncattr('date_time')
    nc.close()

    # Parse datetime
    date_time_clean = date_time.replace('/', '')
    month = int(date_time_clean[4:6])
    hour = int(date_time_clean[8:10])

    # Night filter (18h–06h)
    if not (hour >= 18 or hour <= 6):
        return None, None, None, None

    # Radiance to TB conversion constants
    C1, C2 = 1.19104e-5, 1.43877
    A9, B9, vc9 = 0.9983, 0.627, 930.659
    A4, B4, vc4 = 0.9959, 3.471, 2569.094
    A10, B10, vc10 = 0.9914, 0.408, 839.662

    # Filter fill values
    for ch in [channel_4, channel_9, channel_10]:
        ch[ch == -1000] = np.nan

    TB9 = (C2 * vc9 / np.log((C1 * (vc9**3) / channel_9) + 1) - B9) / A9
    TB4 = (C2 * vc4 / np.log((C1 * (vc4**3) / channel_4) + 1) - B4) / A4
    TB10 = (C2 * vc10 / np.log((C1 * (vc10**3) / channel_10) + 1) - B10) / A10

    red = TB10 - TB9
    green = TB9 - TB4
    blue = TB9

    # FLS detection condition
    fls_mask = (green >= 4.5) & (green <= 9) & (red >= -2) & (red <= 2) & (blue >= 275) & (blue <= 293)

    # Flip for geographic orientation
    fls_mask = np.flipud(fls_mask)
    latitudes = latitudes[::-1]

    return fls_mask.astype(np.uint8), latitudes, longitudes, (month, hour)

# --- Loop through files and aggregate ---
input_dir = "data/image_sat"
output_dir = "data/synthese"
os.makedirs(output_dir, exist_ok=True)

FLS_data = {}

for file in os.listdir(input_dir):
    if not file.endswith(".nc"):
        continue

    fls_mask, lats, lons, time_info = extract_FLS_maroc(os.path.join(input_dir, file))
    if fls_mask is None:
        continue

    month, hour = time_info
    key = f"{month:02d}{hour:02d}"

    if key not in FLS_data:
        FLS_data[key] = {
            "sum": np.zeros_like(fls_mask, dtype=np.int32),
            "count": np.zeros_like(fls_mask, dtype=np.int32),
            "lats": lats,
            "lons": lons
        }

    FLS_data[key]["sum"] += fls_mask
    FLS_data[key]["count"] += (~np.isnan(fls_mask)).astype(int)

# --- Generate MMhh.txt files ---
for key, data in FLS_data.items():
    sum_mask = data["sum"]
    count_mask = data["count"]
    lats, lons = data["lats"], data["lons"]

    freq = np.divide(sum_mask, count_mask, out=np.zeros_like(sum_mask, dtype=float), where=count_mask > 0)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Flatten arrays
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    freq_flat = freq.flatten()
    obs_flat = count_mask.flatten()

    # Keep only points inside Morocco shapefile
    inside_mask = [morocco_geom.contains(Point(lon_flat[i], lat_flat[i])) for i in range(len(lat_flat))]

    df = pd.DataFrame({
        "Latitude": lat_flat[inside_mask],
        "Longitude": lon_flat[inside_mask],
        "Frequence": freq_flat[inside_mask],
        "Total_Obs": obs_flat[inside_mask]
    })

    # Filter out points with 0 observation
    df = df[df["Total_Obs"] > 0]

    txt_path = os.path.join(output_dir, f"{key}.txt")
    df.to_csv(txt_path, sep="\t", index=False, float_format="%.6f")
    print(f" Saved {txt_path}")

print("✅ All MMhh.txt files generated successfully (inside Morocco only).")
