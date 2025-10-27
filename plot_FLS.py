import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap

# === Directories ===
input_dir = "C:/Users/moham/OneDrive/Documents/fog/synthese/synthese"      # Location of MMhh.txt files
output_dir = "C:/Users/moham/OneDrive/Documents/fog"
os.makedirs(output_dir, exist_ok=True)

# === Load Morocco shapefile ===
shapefile_path = "C:/Users/moham/OneDrive/Documents/fog/Morocco_ADM0_simplified.simplified.geojson"
gdf = gpd.read_file(shapefile_path)

# === Custom colormap similar to your example ===
# Black (0) → Deep blue → Light blue → Pale yellow → Orange → Red → Dark red (65%)
colors = [
    (0.0,  'black'),
    (0.05, 'midnightblue'),
    (0.15, 'blue'),
    (0.30, 'lightskyblue'),
    (0.45, 'lemonchiffon'),
    (0.60, 'orange'),
    (0.80, 'red'),
    (1.0,  'darkred')
]
cmap_custom = LinearSegmentedColormap.from_list("black_blue_yellow_red", [c[1] for c in colors], N=256)

# === Function: read and convert a text file into a grid ===
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


# === Function: plot a frequency map ===
def plot_frequency_map(lon_grid, lat_grid, freq_grid, title, output_name):
    fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Convert frequency values to percentage (0–100%)
    freq_percentage = freq_grid * 100
    
    # Plot with custom color scale, but colorbar scaled up to look like 0–60%
    im = ax.pcolormesh(lon_grid, lat_grid, freq_percentage,
                       cmap=cmap_custom,
                       transform=ccrs.PlateCarree(),
                       vmin=0, vmax=20)  # Actual data range (up to 20%)

    # Colorbar: scaled visually as if 0–60%
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", label="FLS Frequency (%)")
    cbar_ticks = np.arange(0, 21, 2)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{t:.0f}%" for t in cbar_ticks])

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)

    plt.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved: {output_name}")


# === Step 1: individual maps for each MMhh.txt file ===
file_list = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
""""
for file in file_list:
    path = os.path.join(input_dir, file)
    lon_grid, lat_grid, freq_grid = read_txt_to_grid(path)
    if freq_grid is not None:
        plot_frequency_map(lon_grid, lat_grid, freq_grid,
                           f"FLS Frequency — {file.replace('.txt', '')}",
                           f"{file.replace('.txt', '')}_plot.png")

# === Step 2: cumulative maps by month and by hour ===
months = sorted(list(set([f[:2] for f in file_list])))  # extract MM from filename "MMhh.txt"
for hour in ["18", "19", "20", "21", "22", "23", "00", "01", "02", "03", "04", "05", "06"]:
    hour_all = sorted(list(set([os.path.join(input_dir, f) for f in file_list if f[2:4] == hour])))
    grids = []
    for fpath in hour_all:
        lon_grid, lat_grid, freq_grid = read_txt_to_grid(fpath)
        if freq_grid is not None:
            grids.append(freq_grid)
    if grids:
        mean_freq = np.nanmean(grids, axis=0)  # hourly average across six months
        plot_frequency_map(lon_grid, lat_grid, mean_freq,
                           f"Average FLS frequency over six months (January–June 2024) for hour {hour}",
                           f"all_{hour}_plot.png")

for month in months:
    month_files = [os.path.join(input_dir, f) for f in file_list if f.startswith(month)]
    grids = []
    for fpath in month_files:
        lon_grid, lat_grid, freq_grid = read_txt_to_grid(fpath)
        if freq_grid is not None:
            grids.append(freq_grid)

    if grids:
        mean_freq = np.nanmean(grids, axis=0)  # monthly average
        plot_frequency_map(lon_grid, lat_grid, mean_freq,
                           f"Monthly mean FLS frequency — Month {month}",
                           f"Month_{month}_plot.png")
"""
# === Step 3: overall map for the full period ===
all_grids = []
for file in file_list:
    lon_grid, lat_grid, freq_grid = read_txt_to_grid(os.path.join(input_dir, file))
    if freq_grid is not None:
        all_grids.append(freq_grid)

if all_grids:
    total_freq = np.nanmean(all_grids, axis=0)  # mean over entire period
    plot_frequency_map(lon_grid, lat_grid, total_freq,
                       "FLS Frequency — Full period (January–December 2024, 18h–6h)",
                       "Total_Period_plot.png")

print(" All FLS maps (hourly, monthly, total period) have been generated with new color scale!")
