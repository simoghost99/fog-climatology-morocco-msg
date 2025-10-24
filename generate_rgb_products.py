import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
from shapely.geometry import Point
import rasterio
from rasterio import features
from rasterio.transform import from_bounds


def plot_night_rgb(
    nc_file,
    add_borders=True,
    use_dynamic_range=True,
    apply_gamma=False,
    gamma=0.5,
    equalize_hist=False,
):
    """Plot Nighttime RGB (Fog Product) over Morocco only."""

    # --- Open NetCDF file ---
    nc = Dataset(nc_file, "r")

    # --- Extract channels and geolocation ---
    channel_4 = nc.variables["channel_4"][:]  # IR3.9
    channel_9 = nc.variables["channel_9"][:]  # IR10.8
    channel_10 = nc.variables["channel_10"][:]  # IR12.0
    latitudes = nc.variables["lat"][:]
    longitudes = nc.variables["lon"][:]

    # --- Extract date/time attribute ---
    date_time = nc.getncattr("date_time")  # e.g., "20240202/04:12"
    date_time_clean = date_time.replace("/", "")  # "202402020412"
    date_time_short = date_time_clean[:10]  # "2024020204"
    output_file = f"data/images/{date_time_short}_RGB_FOG_Morocco.png"

    # --- Check if time is between 18:00-06:00 UTC ---
    hour = int(date_time_clean[8:10])
    if not (hour >= 18 or hour < 6):
        print(f" {nc_file} ignored — not in nighttime period (18:00–06:00 UTC).")
        nc.close()
        return
    nc.close()

    # --- Handle missing values ---
    for ch in [channel_4, channel_9, channel_10]:
        ch[ch == -1000] = np.nan

    # --- Constants for brightness temperature ---
    C1, C2 = 1.19104e-5, 1.43877
    A9, B9, vc9 = 0.9983, 0.627, 930.659  # IR10.8
    A4, B4, vc4 = 0.9959, 3.471, 2569.094  # IR3.9
    A10, B10, vc10 = 0.9914, 0.408, 839.662  # IR12.0

    # --- Compute brightness temperatures ---
    TB9 = (C2 * vc9 / np.log((C1 * (vc9**3) / channel_9) + 1) - B9) / A9
    TB4 = (C2 * vc4 / np.log((C1 * (vc4**3) / channel_4) + 1) - B4) / A4
    TB10 = (C2 * vc10 / np.log((C1 * (vc10**3) / channel_10) + 1) - B10) / A10

    # --- Compute RGB components ---
    red = TB10 - TB9
    green = TB9 - TB4
    blue = TB9

    # --- Dynamic or fixed stretch ---
    if use_dynamic_range:
        red_min, red_max = np.nanpercentile(red, [1, 99])
        green_min, green_max = np.nanpercentile(green, [1, 99])
        blue_min, blue_max = np.nanpercentile(blue, [1, 99])
    else:
        red_min, red_max = -10, 5
        green_min, green_max = -20, 15
        blue_min, blue_max = 210, 300

    # --- Scale to 0–1 range ---
    def scale(ch, vmin, vmax):
        return np.clip((ch - vmin) / (vmax - vmin), 0, 1)

    red_s = scale(red, red_min, red_max)
    green_s = scale(green, green_min, green_max)
    blue_s = scale(blue, blue_min, blue_max)

    rgb_image = np.stack([red_s, green_s, blue_s], axis=-1)
    rgb_image = np.flipud(rgb_image)
    latitudes = latitudes[::-1]

    # --- Convert to 0–255 uint8 for OpenCV ---
    rgb_image_cv = (rgb_image * 255).astype(np.uint8)

    # --- Apply gamma correction ---
    if apply_gamma:
        rgb_image_cv = np.power(rgb_image_cv / 255.0, gamma)
        rgb_image_cv = (rgb_image_cv * 255).astype(np.uint8)

    # --- Apply histogram equalization ---
    if equalize_hist:
        for i in range(3):
            rgb_image_cv[:, :, i] = cv2.equalizeHist(rgb_image_cv[:, :, i])

    # --- Load Morocco shapefile ---
    shapefile_path = (
        "data/maroc shape file/Morocco_ADM0_simplified.simplified(1).geojson"
    )
    gdf = gpd.read_file(shapefile_path)
    morocco_geom = gdf.geometry.unary_union  # merge all polygons

    # --- Create raster mask of Morocco ---
    transform = from_bounds(
        longitudes.min(),
        latitudes.min(),
        longitudes.max(),
        latitudes.max(),
        rgb_image.shape[1],
        rgb_image.shape[0],
    )

    mask = features.rasterize(
        [(morocco_geom, 1)],
        out_shape=(rgb_image.shape[0], rgb_image.shape[1]),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # --- Apply mask (keep only Morocco area) ---
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    rgb_image_cv = np.where(mask_3d == 1, rgb_image_cv, 255)  # White outside Morocco

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.imshow(
        rgb_image_cv,
        extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()],
        transform=ccrs.PlateCarree(),
        origin="upper",
    )
    ax.set_title(f"Nighttime RGB (Fog) — {date_time_short} UTC", fontsize=12)

    # --- Add borders ---
    if add_borders:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
        gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # --- Save output ---
    plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=False)
    plt.close()

    print(f" Image saved: {output_file}")


# === Main execution ===
if __name__ == "__main__":
    input_dir = "data/image_sat"
    nc_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".nc")
    ]

    for file in nc_files:
        print(f"🛰 Processing: {file}")
        try:
            plot_night_rgb(
                file,
                add_borders=True,
                use_dynamic_range=True,
                apply_gamma=True,
                gamma=0.9,
                equalize_hist=False,
            )
        except Exception as e:
            print(f" Error with {file}: {e}")

    print("Processing completed!")
