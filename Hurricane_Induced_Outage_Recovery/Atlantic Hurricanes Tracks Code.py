




import geopandas as gpd
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.colors import qualitative

# ----------------------------------------------------
# Step 1: Set file paths and CRS
# ----------------------------------------------------
hurricane_shapefile_dir = 'C:/Users/aas0041/Downloads/Hurricanes Path'
county_shapefile_path = 'C:/Users/aas0041/Downloads/tl_2023_us_county/tl_2023_us_county.shp'
state_shapefile_path = 'C:/Users/aas0041/Downloads/tl_2023_us_state/tl_2023_us_state.shp'
target_crs = "EPSG:4326"

# ----------------------------------------------------
# Step 2: Load counties and states, filter by selected states
# ----------------------------------------------------
counties = gpd.read_file(county_shapefile_path).to_crs(target_crs)
states = gpd.read_file(state_shapefile_path).to_crs(target_crs)

selected_states = [
    'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont',
    'New Jersey', 'New York', 'Pennsylvania', 'Delaware', 'Florida', 'Georgia', 'Maryland',
    'North Carolina', 'South Carolina', 'Virginia', 'West Virginia',
    'Alabama', 'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas', 'Louisiana', 'Oklahoma', 'Texas',
    'Indiana', 'Ohio', 'Missouri', 'Illinois'
]
states = states[states['NAME'].isin(selected_states)]

# ----------------------------------------------------
# Step 3: Set hurricane code-to-name mapping
# ----------------------------------------------------
code_to_name = {
    "AL092017": "HARVEY", "AL112017": "IRMA", "AL152017": "MARIA", "AL062018": "FLORENCE",
    "AL142018": "MICHAEL", "AL052019": "DORIAN", "AL112019": "IMELDA", "AL082020": "HANNA",
    "AL092020": "ISAIAS", "AL132020": "LAURA", "AL192020": "SALLY", "AL262020": "DELTA",
    "AL282020": "ZETA", "AL092021": "IDA", "AL142021": "NICHOLAS", "AL092022": "IAN",
    "AL172022": "NICOLE", "AL102023": "IDALIA",
    "AL022024": "BERYL", "AL042024": "DEBBY", "AL062024": "FRANCINE", "AL092024": "HELENE" , "AL142024": "MILTON"
}

# ----------------------------------------------------
# Step 4: Load hurricane shapefiles and assign names
# ----------------------------------------------------
hurricane_files = [f for f in os.listdir(hurricane_shapefile_dir) if f.endswith('.shp')]
hurricane_gdfs = []

for filename in sorted(hurricane_files):
    code = os.path.splitext(filename)[0].split('_')[0].upper()
    name = code_to_name.get(code, "UNKNOWN")
    display_name = f"{code} – {name}"

    base_path = os.path.join(hurricane_shapefile_dir, os.path.splitext(filename)[0])
    if not os.path.exists(base_path + ".shx") or not os.path.exists(base_path + ".dbf"):
        print(f"Skipping {filename} — missing files.")
        continue

    gdf = gpd.read_file(base_path + ".shp").to_crs(target_crs)
    gdf["hurricane_name"] = display_name
    hurricane_gdfs.append(gdf)

# Combine all hurricanes into a single DataFrame
all_tracks = pd.concat(hurricane_gdfs, ignore_index=True)

# ----------------------------------------------------
# Step 5: Assign Plotly colors for hurricanes
# ----------------------------------------------------
unique_names = sorted(all_tracks["hurricane_name"].unique())
palette = qualitative.Dark24
color_dict = {name: palette[i % len(palette)] for i, name in enumerate(unique_names)}

# ----------------------------------------------------
# Step 6: Build Plotly figure and add counties
# ----------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Choroplethmapbox(
    geojson=counties.__geo_interface__,
    locations=[i for i in range(len(counties))],
    z=[1]*len(counties),
    colorscale=[[0, '#E5E5E5'], [1, '#E5E5E5']],
    showscale=False,
    marker_line_width=0.3,
    marker_line_color='black'
))

# ----------------------------------------------------
# Step 7: Add hurricane tracks by color
# ----------------------------------------------------
for name in unique_names:
    subset = all_tracks[all_tracks["hurricane_name"] == name]
    color = color_dict[name]
    for geom in subset.geometry:
        if geom.geom_type == 'LineString':
            lon, lat = list(geom.xy[0]), list(geom.xy[1])
            fig.add_trace(go.Scattermapbox(
                lon=lon,
                lat=lat,
                mode='lines',
                line=dict(width=3, color=color),
                hovertemplate=f"<b>{name}</b><extra></extra>",
                showlegend=False
            ))

# ----------------------------------------------------
# Step 8: Add state boundaries 
# ----------------------------------------------------
for _, row in states.iterrows():
    geometries = [row.geometry] if row.geometry.geom_type == "Polygon" else row.geometry.geoms
    for geom in geometries:
        lon, lat = zip(*geom.exterior.coords)
        fig.add_trace(go.Scattermapbox(
            lon=lon,
            lat=lat,
            mode='lines',
            line=dict(width=1.5, color='black'),
            showlegend=False
        ))

# ----------------------------------------------------
# Step 9: Create custom two-column legend for hurricanes
# ----------------------------------------------------
half = (len(unique_names) + 1) // 2
left_col = unique_names[:half]
right_col = unique_names[half:]

annotations = []
y_start = 0.18
dy = 0.03
x_left = 0.01
x_right = 0.17
separator_x = 0.12

for i, name in enumerate(left_col):
    color = color_dict[name]
    annotations.append(dict(
        x=x_left, y=y_start + i * dy,
        xanchor="left", yanchor="bottom",
        text=f"<span style='color:{color}; font-size:22px'>■</span> <span style='color:#222; font-size:13px'>{name}</span>",
        showarrow=False,
        font=dict(size=10),
        xref="paper", yref="paper"
    ))

for i, name in enumerate(right_col):
    color = color_dict[name]
    annotations.append(dict(
        x=x_right, y=y_start + i * dy,
        xanchor="left", yanchor="bottom",
        text=f"<span style='color:{color}; font-size:22px'>■</span> <span style='color:#222; font-size:13px'>{name}</span>",
        showarrow=False,
        font=dict(size=10),
        xref="paper", yref="paper"
    ))

separator_line = dict(
    type="line",
    x0=separator_x, y0=y_start - 0.015,
    x1=separator_x, y1=y_start + (half * dy) + 0.015,
    xref="paper", yref="paper",
    line=dict(color="black", width=3)
)

# ----------------------------------------------------
# Step 10: Finalize layout and title
# ----------------------------------------------------
fig.update_layout(
    mapbox_style="white-bg",
    mapbox=dict(center={"lat": 30, "lon": -85}, zoom=3),
    margin={"r": 20, "t": 40, "l": 20, "b": 20},
    paper_bgcolor='white',
    plot_bgcolor='white',
    showlegend=False,
    annotations=annotations,
    shapes=[separator_line],
    title="Atlantic Hurricane Tracks 2017 - 2024"
)

# ----------------------------------------------------
# Step 11: Save and show figure
# ----------------------------------------------------
fig.write_html("Atlantic Hurricanes Tracks.html")
fig.write_image("Atlantic Hurricanes Tracks.png", width=1600, height=1000, scale=2)
fig.show()
