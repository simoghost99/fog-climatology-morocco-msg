# FLS Climatology — Fog and Low Stratus Detection over Morocco

## Project Overview
**FLS Climatology** is a scientific workflow designed to detect and analyze **Fog and Low Stratus (FLS)** occurrences over Morocco using **infrared geostationary satellite data** from *EUMETSAT Meteosat Second Generation (MSG)*.

This workflow automatically:
- Processes MSG NetCDF infrared data (IR3.9, IR10.8, IR12.0)  
- Detects FLS pixels during nighttime (18:00–06:00 UTC)  
- Masks the domain to Morocco using shapefiles  
- Generates RGB and FLS detection maps  
- Produces monthly/hourly summary files (`MMHH.txt`)  
- Creates frequency climatology maps for each month and night hour  

---

## Scientific Basis
The detection of **Fog and Low Stratus (FLS)** is based on **brightness temperature differences (BTD)** between MSG infrared channels, following methods from *Cermak & Bendix (2008)* and related studies.

| Channel | Wavelength (µm) | Role |
|----------|------------------|------|
| IR3.9 (Ch4) | 3.9 µm | Sensitive to fog and low clouds at night |
| IR10.8 (Ch9) | 10.8 µm | Standard thermal infrared “window” channel |
| IR12.0 (Ch10) | 12.0 µm | Used to separate low and high clouds |

**FLS Detection Criteria:**

$$
4.5 \le (IR_{10.8} - IR_{3.9}) \le 9 \quad \land \quad -2 \le (IR_{12.0} - IR_{10.8}) \le 2
$$

These thresholds identify pixels with thermal characteristics consistent with fog or low stratus layers during nighttime conditions.

## Repository Structure

FLS_Climatology/
│
├── data/
│ ├── image_sat/ # Raw NetCDF MSG infrared data
│ ├── image_bin/ # Binary RGB / FLS mask images
│ ├── synthese/ # Monthly/hourly summary text files (MMHH.txt)
│ ├── plots_from_txt/ # Frequency maps (monthly, hourly)
│ └── maroc shape file/ # GeoJSON shapefile of Morocco
│
├── scripts/
│ ├── extract_FLS_maroc.py # Core FLS detection and domain masking
│ ├── synthese_FLS.py # Builds MMHH.txt frequency tables
│ ├── plot_frequency_maps.py # Generates frequency climatology maps
│ └── utils/ # Helper functions
│
├── README.md
└── requirements.txt 




## References
- Cermak, J., & Bendix, J. (2008). *A novel approach to fog/low stratus detection using Meteosat 8 data*. Atmospheric Research, 87(3–4), 279–292.  
- Cermak, J., Eastman, R. M., Bendix, J., & Warren, S. G. (2009). *European climatology of fog and low stratus based on geostationary satellite observations*. QJRMS, 135(645), 2125–2130.  
- Jahani, B., et al. (2025). *Algorithm for continual monitoring of fog based on geostationary satellite imagery*. Atmospheric Measurement Techniques, 18(8), 1927–1941.


## Repository Structure

