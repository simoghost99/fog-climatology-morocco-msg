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

# === Custom colormap similar ===
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


# === Function: plot a frequency map with enhanced styling ===
def plot_frequency_map(lon_grid, lat_grid, freq_grid, title, output_name):
    fig = plt.figure(figsize=(12, 10))  # Increased size for better visualization
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Convert frequency values to percentage (0–100%)
    freq_percentage = freq_grid * 100
    
    # Plot with custom color scale - CHANGED TO 25%
    im = ax.pcolormesh(lon_grid, lat_grid, freq_percentage,
                       cmap=cmap_custom,
                       transform=ccrs.PlateCarree(),
                       vmin=0, vmax=25)  # Changed from 20 to 25

    # Colorbar with enhanced styling - UPDATED FOR 25%
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", label="FLS Frequency (%)", shrink=0.8)
    cbar_ticks = np.arange(0, 26, 5)  # Changed to 0-25 with step 5
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{t:.0f}%" for t in cbar_ticks])
    cbar.ax.tick_params(labelsize=10)

    # Add geographical features with enhanced styling
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')  # Black coastline
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)  # Ocean background
    
    # Add Morocco shapefile
    gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2, transform=ccrs.PlateCarree())

    # Set extended map extent for Morocco
    ax.set_extent([-20, -1, 20, 36], crs=ccrs.PlateCarree())  # Extended bounds
    
    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_xlabel("Longitude", fontsize=12, fontweight='bold')
    ax.set_ylabel("Latitude", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add data source note
    ax.text(0.02, 0.02, "Data: Gridded FLS Frequency\nMap extent: Longitude [-20, -1], Latitude [20, 36]", 
           transform=ax.transAxes, fontsize=8,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved: {output_name}")


# === Function: plot enhanced frequency map with statistics ===
def plot_enhanced_frequency_map(lon_grid, lat_grid, freq_grid, title, output_name):
    """Enhanced version with statistics and better styling"""
    
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Convert frequency values to percentage
    freq_percentage = freq_grid * 100
    
    # Calculate statistics
    valid_data = freq_percentage[~np.isnan(freq_percentage)]
    if len(valid_data) > 0:
        avg_freq = np.mean(valid_data)
        max_freq = np.max(valid_data)
        min_freq = np.min(valid_data)
    else:
        avg_freq = max_freq = min_freq = 0
    
    # Plot with custom color scale - CHANGED TO 25%
    im = ax.pcolormesh(lon_grid, lat_grid, freq_percentage,
                       cmap=cmap_custom,
                       transform=ccrs.PlateCarree(),
                       vmin=0, vmax=25)  # Changed from 20 to 25

    # Enhanced colorbar - UPDATED FOR 25%
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", label="FLS Frequency (%)", shrink=0.8, pad=0.05)
    cbar_ticks = np.arange(0, 26, 5)  # Changed to 0-25 with step 5
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{t:.0f}%" for t in cbar_ticks])
    cbar.ax.tick_params(labelsize=10)

    # Geographical features
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
    
    # Morocco shapefile
    gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5, transform=ccrs.PlateCarree())

    # Set extended map extent
    ax.set_extent([-20, -1, 20, 36], crs=ccrs.PlateCarree())
    
    # Enhanced gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_xlabel("Longitude", fontsize=12, fontweight='bold')
    ax.set_ylabel("Latitude", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Statistics box
    stats_text = f"Statistics:\n"
    stats_text += f"Avg: {avg_freq:.1f}%\n"
    stats_text += f"Max: {max_freq:.1f}%\n"
    stats_text += f"Min: {min_freq:.1f}%\n"
    stats_text += f"Points: {len(valid_data)}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
           verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Enhanced map saved: {output_name}")


# === Process all files ===
file_list = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

# === Individual maps for each file (commented out as before) ===
"""
for file in file_list:
    path = os.path.join(input_dir, file)
    lon_grid, lat_grid, freq_grid = read_txt_to_grid(path)
    if freq_grid is not None:
        plot_frequency_map(lon_grid, lat_grid, freq_grid,
                           f"FLS Frequency — {file.replace('.txt', '')}",
                           f"{file.replace('.txt', '')}_plot.png")
"""

# === Cumulative maps by month and by hour (commented out as before) ===
"""
months = sorted(list(set([f[:2] for f in file_list])))
for hour in ["18", "19", "20", "21", "22", "23", "00", "01", "02", "03", "04", "05", "06"]:
    hour_all = sorted(list(set([os.path.join(input_dir, f) for f in file_list if f[2:4] == hour])))
    grids = []
    for fpath in hour_all:
        lon_grid, lat_grid, freq_grid = read_txt_to_grid(fpath)
        if freq_grid is not None:
            grids.append(freq_grid)
    if grids:
        mean_freq = np.nanmean(grids, axis=0)
        plot_enhanced_frequency_map(lon_grid, lat_grid, mean_freq,
                           f"Average FLS frequency (Hour {hour})",
                           f"enhanced_hour_{hour}_plot.png")

for month in months:
    month_files = [os.path.join(input_dir, f) for f in file_list if f.startswith(month)]
    grids = []
    for fpath in month_files:
        lon_grid, lat_grid, freq_grid = read_txt_to_grid(fpath)
        if freq_grid is not None:
            grids.append(freq_grid)

    if grids:
        mean_freq = np.nanmean(grids, axis=0)
        plot_enhanced_frequency_map(lon_grid, lat_grid, mean_freq,
                           f"Monthly mean FLS frequency — Month {month}",
                           f"enhanced_month_{month}_plot.png")
"""

# === Overall map for the full period ===
print("Processing overall map for full period...")
all_grids = []
for file in file_list:
    lon_grid, lat_grid, freq_grid = read_txt_to_grid(os.path.join(input_dir, file))
    if freq_grid is not None:
        all_grids.append(freq_grid)

if all_grids:
    total_freq = np.nanmean(all_grids, axis=0)  # mean over entire period
    
    # Create both standard and enhanced versions
    plot_frequency_map(lon_grid, lat_grid, total_freq,
                       "FLS Frequency — Full period (January–December 2024, 18h–6h)",
                       "Total_Period_plot.png")
    
    plot_enhanced_frequency_map(lon_grid, lat_grid, total_freq,
                       "FLS Frequency — Full period (January–December 2024, 18h–6h)",
                       "Enhanced_Total_Period_plot.png")

print("\n" + "="*70)
print("ALL FLS MAPS HAVE BEEN GENERATED WITH 25% COLOR SCALE!")
print("Features:")
print("• 25% color scale (updated from 20%)")
print("• Black coastline")
print("• Extended map extent (Longitude: -20 to -1, Latitude: 20 to 36)")
print("• Ocean background")
print("• Clean design without frequency range legend")
print("• Statistics box (enhanced version)")
print("• Professional gridlines")
print("="*70)
