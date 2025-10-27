import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap

# === Directories ===
input_dir = "C:/Users/moham/OneDrive/Documents/fog/synthese/synthese"
output_dir = "C:/Users/moham/OneDrive/Documents/fog"
os.makedirs(output_dir, exist_ok=True)

# === Load Morocco shapefile ===
shapefile_path = "C:/Users/moham/OneDrive/Documents/fog/Morocco_ADM0_simplified.simplified.geojson"
gdf = gpd.read_file(shapefile_path)

# === Custom color palette similar to your reference image ===
# Black → deep blue → light blue → light yellow → orange → red → dark red
colors = [
    (0.0, "black"),       # 0%
    (0.10, "#08306b"),    # deep blue
    (0.25, "#2171b5"),    # medium blue
    (0.35, "#6baed6"),    # light blue
    (0.45, "#fff7bc"),    # pale yellow
    (0.55, "#fdae61"),    # orange
    (0.65, "#f46d43"),    # strong orange-red
    (0.80, "#d73027"),    # red
    (1.0, "#7f0000")      # dark red
]
cmap_custom = LinearSegmentedColormap.from_list("black_blue_yellow_red", [c[1] for c in colors], N=256)

# === Function to read text files into grids ===
def read_txt_to_grid(file_path):
    df = pd.read_csv(file_path, sep="\t")
    if df.empty:
        return None, None, None

    lats = np.unique(df["Latitude"].values)
    lons = np.unique(df["Longitude"].values)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    freq_grid = np.full_like(lon_grid, np.nan, dtype=float)

    for _, row in df.iterrows():
        lat_idx = np.argmin(np.abs(lats - row["Latitude"]))
        lon_idx = np.argmin(np.abs(lons - row["Longitude"]))
        freq_grid[lat_idx, lon_idx] = row["Frequence"]

    return lon_grid, lat_grid, freq_grid


# === Function to plot frequency map ===
def plot_frequency_map(lon_grid, lat_grid, freq_grid, title, output_name):
    fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    freq_percentage = freq_grid * 100  # convert to %
    im = ax.pcolormesh(
        lon_grid, lat_grid, freq_percentage,
        cmap=cmap_custom,
        transform=ccrs.PlateCarree(),
        vmin=0, vmax=65  # capped at 65% (dark red)
    )

    # Add shapefile & coastlines
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
    gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=0.6)

    # === Colorbar ===
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.85, pad=0.02)
    cbar.set_label("FLS Frequency (%)", fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    ticks = np.arange(0, 70, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t}%" for t in ticks])

    # === Labels & Title ===
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title(title, fontsize=13, pad=15, weight="bold")

    plt.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Map saved: {output_name}")


# === Step 1: Individual MMhh maps ===
file_list = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

for file in file_list:
    path = os.path.join(input_dir, file)
    lon_grid, lat_grid, freq_grid = read_txt_to_grid(path)
    if freq_grid is not None:
        plot_frequency_map(
            lon_grid, lat_grid, freq_grid,
            f"FLS Frequency — {file.replace('.txt', '')}",
            f"{file.replace('.txt', '')}_plot.png"
        )

# === Step 2: Hourly averages ===
hours = ["18", "19", "20", "21", "22", "23", "00", "01", "02", "03", "04", "05", "06"]
months = sorted(list(set([f[:2] for f in file_list])))

for hour in hours:
    hour_files = [os.path.join(input_dir, f) for f in file_list if f[2:4] == hour]
    grids = []
    for fpath in hour_files:
        lon_grid, lat_grid, freq_grid = read_txt_to_grid(fpath)
        if freq_grid is not None:
            grids.append(freq_grid)
    if grids:
        mean_freq = np.nanmean(grids, axis=0)
        plot_frequency_map(
            lon_grid, lat_grid, mean_freq,
            f"Average FLS Frequency (Jan–Jun 2024) — Hour {hour} UTC",
            f"AllMonths_Hour_{hour}_plot.png"
        )

# === Step 3: Monthly averages ===
for month in months:
    month_files = [os.path.join(input_dir, f) for f in file_list if f.startswith(month)]
    grids = []
    for fpath in month_files:
        lon_grid, lat_grid, freq_grid = read_txt_to_grid(fpath)
        if freq_grid is not None:
            grids.append(freq_grid)
    if grids:
        mean_freq = np.nanmean(grids, axis=0)
        plot_frequency_map(
            lon_grid, lat_grid, mean_freq,
            f"Monthly Mean FLS Frequency — {month} (2024)",
            f"Month_{month}_plot.png"
        )

# === Step 4: Total period ===
all_grids = []
for file in file_list:
    lon_grid, lat_grid, freq_grid = read_txt_to_grid(os.path.join(input_dir, file))
    if freq_grid is not None:
        all_grids.append(freq_grid)

if all_grids:
    total_freq = np.nanmean(all_grids, axis=0)
    plot_frequency_map(
        lon_grid, lat_grid, total_freq,
        "FLS Frequency — Full Period (Jan–Jun 2024, 18h–06h)",
        "Total_Period_plot.png"
    )

print("\n All FLS maps (hourly, monthly, total) generated successfully with new color scale!")

