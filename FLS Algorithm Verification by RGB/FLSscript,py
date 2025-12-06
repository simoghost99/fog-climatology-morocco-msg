import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import os

# --- Fonction existante (adaptée pour le calcul seul) ---
def calculate_night_rgb(nc_file):
    """
    Calcule les composants RGB du produit Nighttime Microphysics
    et retourne les trois matrices numpy (Red, Green, Blue) ainsi que les coordonnées.
    """
    nc = Dataset(nc_file, "r")
    
    try:
        channel_4 = nc.variables["channel_4"][:]
        channel_9 = nc.variables["channel_9"][:]
        channel_10 = nc.variables["channel_10"][:]
        latitudes = nc.variables["lat"][:]
        longitudes = nc.variables["lon"][:]
        date_time = nc.getncattr("date_time")
        date_time_clean = date_time.replace("/", "")
    except KeyError as e:
        print(f"⚠️ Variable manquante dans {nc_file}: {e}")
        nc.close()
        return None, None, None, None, None, None
        
    nc.close()

    # Gestion des valeurs manquantes
    for ch in [channel_4, channel_9, channel_10]:
        ch[ch == -1000] = np.nan

    # Constantes
    C1, C2 = 1.19104e-5, 1.43877
    A9, B9, vc9 = 0.9983, 0.6084, 929.842    # IR10.8
    A4, B4, vc4 = 0.9915, 2.9002, 2547.771  # IR3.9
    A10, B10, vc10 = 0.9988, 0.3882, 838.659  # IR12.0

    # Calcul des TB
    TB9 = (C2 * vc9 / np.log((C1 * (vc9**3) / channel_9) + 1) - B9) / A9
    TB4 = (C2 * vc4 / np.log((C1 * (vc4**3) / channel_4) + 1) - B4) / A4
    TB10 = (C2 * vc10 / np.log((C1 * (vc10**3) / channel_10) + 1) - B10) / A10

    # Calcul des composants RGB (BTDs)
    red = TB10 - TB9
    green = TB9 - TB4
    blue = TB9
    
    # Inversion des données (si nécessaire, comme dans votre code initial)
    red = np.flipud(red)
    green = np.flipud(green)
    blue = np.flipud(blue)
    latitudes = latitudes[::-1]

    # Convertir en arrays numpy réguliers (pas de MaskedArray)
    red = np.asarray(red)
    green = np.asarray(green)
    blue = np.asarray(blue)

    return red, green, blue, latitudes, longitudes, date_time_clean

# --- Fonction pour créer et visualiser le masque FLS binaire ---
def create_and_plot_fls_mask(nc_file, output_dir="output", 
                           plot_rgb=True, plot_mask=True, 
                           add_features=True,
                           threshold_green=(4.5, 10),
                           threshold_red=(-2, 2),
                           threshold_blue=(275, 293)):
    """
    Crée un masque binaire FLS (Fog/Low Stratus) et le visualise en noir et blanc.
    
    Paramètres:
    -----------
    nc_file : str
        Chemin vers le fichier NetCDF
    output_dir : str
        Répertoire de sortie pour les images
    plot_rgb : bool
        Afficher l'image RGB Nighttime Microphysics
    plot_mask : bool
        Afficher le masque binaire FLS
    add_features : bool
        Ajouter les côtes, frontières, etc.
    threshold_green : tuple
        Seuils pour le canal vert (green_min, green_max)
    threshold_red : tuple
        Seuils pour le canal rouge (red_min, red_max)
    threshold_blue : tuple
        Seuils pour le canal bleu (blue_min, blue_max)
    """
    
    # 1. Calculer les composants RGB
    print(f"📊 Calcul des composants RGB depuis: {nc_file}")
    red, green, blue, latitudes, longitudes, date_time = calculate_night_rgb(nc_file)
    
    if red is None:
        print("❌ Erreur lors du calcul des composants RGB")
        return None
    
    print(f"✅ Données calculées pour: {date_time}")
    print(f"   Forme des données: {red.shape}")
    print(f"   Latitude: {latitudes.min():.2f} à {latitudes.max():.2f}")
    print(f"   Longitude: {longitudes.min():.2f} à {longitudes.max():.2f}")
    print(f"   Valeurs Red: {np.nanmin(red):.2f} à {np.nanmax(red):.2f}")
    print(f"   Valeurs Green: {np.nanmin(green):.2f} à {np.nanmax(green):.2f}")
    print(f"   Valeurs Blue: {np.nanmin(blue):.2f} à {np.nanmax(blue):.2f}")
    
    # 2. Créer le masque FLS binaire
    print("\n🎭 Création du masque FLS binaire...")
    print(f"   Seuils: Green ∈ [{threshold_green[0]}, {threshold_green[1]}]")
    print(f"           Red ∈ [{threshold_red[0]}, {threshold_red[1]}]")
    print(f"           Blue ∈ [{threshold_blue[0]}, {threshold_blue[1]}]")
    
    # Condition FLS - gérer les NaN
    valid_pixels = ~np.isnan(green) & ~np.isnan(red) & ~np.isnan(blue)
    
    # Initialiser le masque avec False
    fls_mask = np.zeros_like(green, dtype=bool)
    
    # Appliquer les conditions seulement aux pixels valides
    fls_mask[valid_pixels] = (
        (green[valid_pixels] >= threshold_green[0]) & 
        (green[valid_pixels] <= threshold_green[1]) & 
        (red[valid_pixels] >= threshold_red[0]) & 
        (red[valid_pixels] <= threshold_red[1]) & 
        (blue[valid_pixels] >= threshold_blue[0]) & 
        (blue[valid_pixels] <= threshold_blue[1])
    )
    
    # Convertir en masque binaire (1 = FLS, 0 = non-FLS)
    fls_binary = fls_mask.astype(np.uint8)
    
    # Statistiques du masque
    total_pixels = fls_binary.size
    fls_pixels = np.sum(fls_binary)
    fls_percentage = (fls_pixels / total_pixels) * 100
    
    print(f"\n📈 Statistiques du masque FLS:")
    print(f"   Pixels totaux: {total_pixels}")
    print(f"   Pixels valides (non-NaN): {np.sum(valid_pixels)}")
    print(f"   Pixels FLS: {fls_pixels} ({fls_percentage:.2f}%)")
    print(f"   Pixels non-FLS: {total_pixels - fls_pixels}")
    
    # 3. Créer des figures
    if plot_rgb and plot_mask:
        fig = plt.figure(figsize=(15, 6))
        n_subplots = 2
    else:
        fig = plt.figure(figsize=(8, 8))
        n_subplots = 1
    
    # Créer l'étendue géographique
    if isinstance(longitudes, np.ndarray) and isinstance(latitudes, np.ndarray):
        if longitudes.ndim == 1 and latitudes.ndim == 1:
            domain_extent = [longitudes.min(), longitudes.max(), 
                            latitudes.min(), latitudes.max()]
        else:
            domain_extent = [np.nanmin(longitudes), np.nanmax(longitudes),
                            np.nanmin(latitudes), np.nanmax(latitudes)]
    else:
        print("⚠️ Coordonnées non valides, utilisation d'une étendue par défaut")
        domain_extent = [-15, 0, 20, 40]  # Maroc par défaut
    
    # 3a. Plot RGB Nighttime Microphysics (optionnel)
    if plot_rgb:
        ax1 = plt.subplot(1, n_subplots, 1, projection=ccrs.PlateCarree())
        ax1.set_extent(domain_extent, crs=ccrs.PlateCarree())
        
        # Normaliser les canaux pour affichage RGB
        def normalize_channel(channel):
            # Utiliser les percentiles pour éviter les valeurs extrêmes
            vmin, vmax = np.nanpercentile(channel, [1, 99])
            channel_norm = np.clip((channel - vmin) / (vmax - vmin), 0, 1)
            return np.nan_to_num(channel_norm, nan=0)  # Remplacer NaN par 0
        
        red_norm = normalize_channel(red)
        green_norm = normalize_channel(green)
        blue_norm = normalize_channel(blue)
        
        rgb_image = np.dstack([red_norm, green_norm, blue_norm])
        
        ax1.imshow(rgb_image,
                  extent=domain_extent,
                  transform=ccrs.PlateCarree(),
                  origin='upper',
                  interpolation='nearest')
        
        ax1.set_title(f"Nighttime Microphysics RGB\n{date_time}", fontsize=12)
        
        if add_features:
            ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="white")
            ax1.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor="white", alpha=0.7)
        
        ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                     alpha=0.3, linewidth=0.5, color='white')
        
    # 3b. Plot masque FLS binaire (NOIR et BLANC)
    if plot_mask:
        if plot_rgb:
            ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
        else:
            ax2 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        ax2.set_extent(domain_extent, crs=ccrs.PlateCarree())
        
        # Créer une image avec 3 canaux pour affichage noir/blanc
        # Blanc = FLS (1), Noir = non-FLS (0)
        mask_display = np.zeros((*fls_binary.shape, 3), dtype=np.uint8)
        
        # Pixels FLS en blanc (255,255,255)
        mask_display[fls_binary == 1] = [255, 255, 255]
        
        # Pixels non-FLS en noir (0,0,0)
        mask_display[fls_binary == 0] = [0, 0, 0]
        
        ax2.imshow(mask_display,
                  extent=domain_extent,
                  transform=ccrs.PlateCarree(),
                  origin='upper',
                  interpolation='nearest')
        
        ax2.set_title(f"Masque FLS Binaire (Noir/Blanc)\n{date_time}", fontsize=12)
        
        if add_features:
            # Ajouter les côtes et frontières en rouge pour meilleure visibilité
            ax2.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor="red")
            ax2.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="red", alpha=0.8)
        
        # Ajouter une légende simple
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label=f'FLS ({fls_pixels} px, {fls_percentage:.1f}%)'),
            Patch(facecolor='black', edgecolor='black', label=f'Non-FLS ({total_pixels - fls_pixels} px)')
        ]
        ax2.legend(handles=legend_elements, loc='lower left', fontsize=8, 
                  facecolor='lightgray', framealpha=0.8)
        
        # Gridlines uniquement si pas de RGB
        if not plot_rgb:
            ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                         alpha=0.3, linewidth=0.5, color='red')
            ax2.set_xlabel("Longitude", fontsize=10)
            ax2.set_ylabel("Latitude", fontsize=10)
    
    plt.tight_layout()
    
    # 4. Sauvegarder les résultats
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder l'image
    output_file = os.path.join(output_dir, f"fls_mask_{date_time}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Image sauvegardée: {output_file}")
    
    # Sauvegarder le masque binaire en format numpy (comme array régulier)
    mask_file = os.path.join(output_dir, f"fls_mask_{date_time}.npy")
    
    # Convertir en array numpy régulier avant de sauvegarder
    fls_binary_regular = np.asarray(fls_binary)
    np.save(mask_file, fls_binary_regular)
    print(f"✅ Masque binaire sauvegardé: {mask_file}")
    
    # Sauvegarder les statistiques
    stats_file = os.path.join(output_dir, f"fls_stats_{date_time}.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Date: {date_time}\n")
        f.write(f"Fichier source: {nc_file}\n")
        f.write(f"Dimensions: {fls_binary.shape[0]} x {fls_binary.shape[1]}\n")
        f.write(f"Pixels totaux: {total_pixels}\n")
        f.write(f"Pixels valides (non-NaN): {np.sum(valid_pixels)}\n")
        f.write(f"Pixels FLS: {fls_pixels} ({fls_percentage:.2f}%)\n")
        f.write(f"Pixels non-FLS: {total_pixels - fls_pixels}\n")
        f.write(f"\nSeuils utilisés:\n")
        f.write(f"  Green: [{threshold_green[0]}, {threshold_green[1]}]\n")
        f.write(f"  Red: [{threshold_red[0]}, {threshold_red[1]}]\n")
        f.write(f"  Blue: [{threshold_blue[0]}, {threshold_blue[1]}]\n")
        f.write(f"\nPlages des données:\n")
        f.write(f"  Green: min={np.nanmin(green):.2f}, max={np.nanmax(green):.2f}\n")
        f.write(f"  Red: min={np.nanmin(red):.2f}, max={np.nanmax(red):.2f}\n")
        f.write(f"  Blue: min={np.nanmin(blue):.2f}, max={np.nanmax(blue):.2f}\n")
    
    print(f"✅ Statistiques sauvegardées: {stats_file}")
    
    # 5. Afficher un aperçu supplémentaire des valeurs
    print("\n📋 Aperçu des valeurs seuils vs données réelles:")
    print(f"   Green - Seuils: [{threshold_green[0]}, {threshold_green[1]}]")
    print(f"           Données: {np.nanpercentile(green, [5, 50, 95]):.1f} (5%, 50%, 95%)")
    
    print(f"   Red - Seuils: [{threshold_red[0]}, {threshold_red[1]}]")
    print(f"         Données: {np.nanpercentile(red, [5, 50, 95]):.1f} (5%, 50%, 95%)")
    
    print(f"   Blue - Seuils: [{threshold_blue[0]}, {threshold_blue[1]}]")
    print(f"          Données: {np.nanpercentile(blue, [5, 50, 95]):.1f} (5%, 50%, 95%)")
    
    # 6. Afficher la figure
    plt.show()
    
    # 7. Retourner les résultats
    return {
        'red': red,
        'green': green,
        'blue': blue,
        'latitudes': latitudes,
        'longitudes': longitudes,
        'date_time': date_time,
        'fls_mask': fls_mask,
        'fls_binary': fls_binary,
        'valid_pixels': valid_pixels,
        'stats': {
            'total_pixels': total_pixels,
            'valid_pixels': np.sum(valid_pixels),
            'fls_pixels': fls_pixels,
            'fls_percentage': fls_percentage
        }
    }

# --- Fonction simple pour afficher seulement le masque binaire ---
def plot_simple_binary_mask(nc_file, output_dir="output_simple"):
    """
    Version simplifiée pour afficher uniquement le masque binaire.
    """
    print(f"🎭 Création du masque FLS binaire simple...")
    
    # Calculer les composants RGB
    red, green, blue, latitudes, longitudes, date_time = calculate_night_rgb(nc_file)
    
    if red is None:
        return None
    
    # Créer le masque FLS
    valid_pixels = ~np.isnan(green) & ~np.isnan(red) & ~np.isnan(blue)
    
    fls_mask = np.zeros_like(green, dtype=bool)
    fls_mask[valid_pixels] = (
        (green[valid_pixels] >= 4.5) & 
        (green[valid_pixels] <= 10) & 
        (red[valid_pixels] >= -2) & 
        (red[valid_pixels] <= 2) & 
        (blue[valid_pixels] >= 275) & 
        (blue[valid_pixels] <= 293)
    )
    
    fls_binary = fls_mask.astype(np.uint8)
    
    # Créer la figure
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Définir l'étendue
    if longitudes.ndim == 1 and latitudes.ndim == 1:
        extent = [longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()]
    else:
        extent = [np.nanmin(longitudes), np.nanmax(longitudes),
                 np.nanmin(latitudes), np.nanmax(latitudes)]
    
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Créer l'image noir/blanc
    mask_display = np.zeros((*fls_binary.shape, 3), dtype=np.uint8)
    mask_display[fls_binary == 1] = [255, 255, 255]  # Blanc pour FLS
    mask_display[fls_binary == 0] = [0, 0, 0]        # Noir pour non-FLS
    
    ax.imshow(mask_display,
              extent=extent,
              transform=ccrs.PlateCarree(),
              origin='upper',
              interpolation='nearest')
    
    # Ajouter les caractéristiques géographiques
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor="red")
    
    ax.set_title(f"Masque FLS Binaire\n{date_time}", fontsize=14)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                alpha=0.3, linewidth=0.5, color='red')
    
    # Statistiques
    total_pixels = fls_binary.size
    fls_pixels = np.sum(fls_binary)
    fls_percentage = (fls_pixels / total_pixels) * 100
    
    # Ajouter une légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='FLS'),
        Patch(facecolor='black', edgecolor='black', label='Non-FLS'),
        Patch(facecolor='none', edgecolor='red', label=f'FLS: {fls_pixels} px ({fls_percentage:.1f}%)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
    
    plt.tight_layout()
    
    # Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"binary_mask_{date_time}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Masque binaire sauvegardé: {output_file}")
    
    plt.show()
    
    return fls_binary

# --- Exemple d'utilisation principale ---
if __name__ == "__main__":
    # Exemple de fichier NetCDF
    nc_file = "votre_fichier.nc"
    
    print("=" * 60)
    print("CRÉATION DU MASQUE FLS BINAIRE (NOIR ET BLANC)")
    print("=" * 60)
    for nc_file in nc_files_to_process:
    # Option 1: Version complète avec RGB et masque
        plot_simple_binary_mask(nc_file)
