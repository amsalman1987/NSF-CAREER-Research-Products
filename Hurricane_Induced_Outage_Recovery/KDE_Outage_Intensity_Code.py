



import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib.path import Path

# ---------------------------------------------------------
# Step 1: Load county shapefile and reproject
# ---------------------------------------------------------
counties = (
    gpd.read_file(
        'C:/Users/aas0041/Documents/ArcGIS/Projects/National structures/tl_rd22_us_county.shp'
    )
    .to_crs(epsg=5070)
)

# ---------------------------------------------------------
# Step 2: Load Excel data with outage information
# ---------------------------------------------------------
data = pd.read_excel(
    'C:/Users/aas0041/Desktop/eaglei_outages/02_2022_Combined_Hurricane_Data_Normalized.xlsx'
)

# ---------------------------------------------------------
# Step 3: Ensure 'FIPS' columns match types for merging
# ---------------------------------------------------------
counties['FIPS'] = counties['FIPS'].astype(str)
data['FIPS']     = data['FIPS'].astype(str)

# ---------------------------------------------------------
# Step 4: Merge datasets on 'FIPS'
# ---------------------------------------------------------
counties = counties.merge(data, on='FIPS')

# ---------------------------------------------------------
# Step 5: Filter to target counties (based on criteria)
# ---------------------------------------------------------
counties_subset = counties[
    (counties['max_percent_outage']     >= 5)   &
    (counties['max_percent_outage']     <= 100) &
    (counties['total22_customers']      >= 1)   &
    (counties['recovery_time_minutes']  >= 60)
].copy()

# ---------------------------------------------------------
# Step 6: Calculate county centroids for KDE
# ---------------------------------------------------------
counties_subset.loc[:, 'centroid'] = counties_subset.geometry.centroid
x = counties_subset['centroid'].x
y = counties_subset['centroid'].y

# ---------------------------------------------------------
# Step 7: Perform KDE (weighted by max_percent_outage)
# ---------------------------------------------------------
xy = np.vstack([x, y])
kde = gaussian_kde(xy, weights=counties_subset['max_percent_outage'])
xmin, ymin, xmax, ymax = counties_subset.total_bounds
xi, yi = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

# ---------------------------------------------------------
# Step 8: Load and reproject state shapefile
# ---------------------------------------------------------
states = (
    gpd.read_file(
        'C:/Users/aas0041/Downloads/tl_2023_us_state/tl_2023_us_state.shp'
    )
    .to_crs(epsg=5070)
)

# ---------------------------------------------------------
# Step 9: Filter relevant states & calculate centroids
# ---------------------------------------------------------
relevant_state_fps = counties_subset['STATEFP'].unique()
states_relevant = states[states['STATEFP'].isin(relevant_state_fps)].copy()
states_relevant.loc[:, 'centroid'] = states_relevant.geometry.centroid

# ---------------------------------------------------------
# Step 10: Plot the county boundaries, KDE, and state boundaries
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

# County boundaries (all)
counties.boundary.plot(ax=ax, linewidth=0.3, color='gray')

# KDE heatmap
heatmap = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Reds')
plt.colorbar(heatmap, label='Outage Intensity')

# State boundaries (bold)
states_relevant.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

# ---------------------------------------------------------
# Step 11: Clip the KDE heatmap to the state borders
# ---------------------------------------------------------
state_union = states_relevant.unary_union
if state_union.geom_type == 'MultiPolygon':
    polys = state_union.geoms
else:
    polys = [state_union]

paths = [Path(np.array(poly.exterior.coords)) for poly in polys]
compound_path = Path.make_compound_path(*paths)
heatmap.set_clip_path(compound_path, transform=ax.transData)

# ---------------------------------------------------------
# Step 12: Annotate state abbreviations at centroid
# ---------------------------------------------------------
for _, row in states_relevant.iterrows():
    ax.text(
        row['centroid'].x,
        row['centroid'].y,
        row['STUSPS'],
        fontsize=8,
        ha='center',
        va='center',
        color='blue'
    )

ax.set_title('KDE of Outage Intensity')
ax.axis('off')

# ---------------------------------------------------------
# Step 13: Save the plot to file
# ---------------------------------------------------------
plt.savefig('KDE_Outage_Intensity.png', dpi=300, bbox_inches='tight')
plt.show()
