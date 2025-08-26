


# SELECT STATES AFFECTED BY HURRICANE

# -------------------------------
# Step 1: Import required libraries
# -------------------------------
import pandas as pd

# -------------------------------
# Step 2: Load the CSV files into DataFrames
# -------------------------------
file1 = pd.read_csv('C:/Users/aas0041/Desktop/eaglei_outages/eaglei_outages_2024.csv', encoding='ISO-8859-1')
file2 = pd.read_csv('C:/Users/aas0041/Desktop/eaglei_outages/mcc_output.csv', encoding='ISO-8859-1')

# -------------------------------
# Step 3: Convert the date column to pandas datetime
# -------------------------------
file1['run_start_time'] = pd.to_datetime(file1['run_start_time'])  

# -------------------------------
# Step 4: Specify the date range for filtering
# -------------------------------
start_date = pd.Timestamp("2024-09-25")
end_date = pd.Timestamp("2024-10-18")

# -------------------------------
# Step 5: Filter file1 for the date range
# -------------------------------
file1 = file1[(file1['run_start_time'] >= start_date) & (file1['run_start_time'] <= end_date)]

# -------------------------------
# Step 6: Specify the states or regions to include
# -------------------------------
selected_states = [
    'Florida', 'Georgia', 'Illinois', 'Indiana', 'Kentucky',
    'North Carolina', 'Ohio', 'South Carolina',
    'Tennessee', 'Virginia', 'West Virginia'
]

# -------------------------------
# Step 7: Filter file1 for the selected states
# -------------------------------
file1 = file1[file1['state'].isin(selected_states)] 

print(f" Filtered records: {len(file1)}")

# -------------------------------
# Step 8: Merge file1 and file2 on FIPS code
# -------------------------------
merged_df = pd.merge(
    file1,
    file2[['FIPS', 'total22_customers']],
    left_on='fips_code',   # Make sure this matches your column names!
    right_on='FIPS',
    how='left'
)

# -------------------------------
# Step 9: Save the merged DataFrame to a new CSV file
# -------------------------------
merged_df.to_csv('Helene22xCustm.csv', index=False)

print(" Merge completed and saved as 'Helene22xCustm.csv'")












# CALCULATE PERCENTAGE OUTAGE AND RECOVERY TIME FOR THE SELECTED COUNTIES


# -------------------------------
# Step 1: Import required libraries
# -------------------------------
import pandas as pd

# -------------------------------
# Step 2: Define the file path and load CSV
# -------------------------------
file_path = 'C:/Users/aas0041/Desktop/eaglei_outages/Helene22xCustm.csv'
df = pd.read_csv(file_path)

# -------------------------------
# Step 3: Rename columns for clarity
# -------------------------------
df.rename(columns={'run_start_time': '15mins_time', 'customers_out': 'sum_max'}, inplace=True)

# -------------------------------
# Step 4: Ensure '15mins_time' is in datetime format
# -------------------------------
df['15mins_time'] = pd.to_datetime(df['15mins_time'])

# -------------------------------
# Step 5: Calculate percentage outage
# -------------------------------
df['PercentageOutage'] = df['sum_max'] / df['total22_customers'] * 100

# -------------------------------
# Step 6: Define function to calculate recovery times per FIPS group
# -------------------------------
def calculate_recovery_times(group):
    if group['PercentageOutage'].isna().all():
        return pd.Series({
            'recovery_time_minutes': None,
            'recovery_time_hours': None,
            'recovery_time_days': None,
            'max_date': None,
            'max_percent_outage': None,
            'sum_max_at_max_date': None,
            'below_5_date': None,
            'below_5_percent_outage': None,
            'sum_max_at_below_5_date': None
        })

    max_index = group['PercentageOutage'].idxmax()
    if pd.isna(max_index):
        return pd.Series({
            'recovery_time_minutes': None,
            'recovery_time_hours': None,
            'recovery_time_days': None,
            'max_date': None,
            'max_percent_outage': None,
            'sum_max_at_max_date': None,
            'below_5_date': None,
            'below_5_percent_outage': None,
            'sum_max_at_below_5_date': None
        })

    max_date = group.loc[max_index, '15mins_time']
    max_percentage_outage = group.loc[max_index, 'PercentageOutage']
    sum_max_at_max_date = group.loc[max_index, 'sum_max']

    subset = group.loc[max_index:]
    below_5_index = subset[subset['PercentageOutage'] < 5].index.min()

    if pd.notna(below_5_index):
        below_5_date = group.loc[below_5_index, '15mins_time']
        below_5_percentage_outage = group.loc[below_5_index, 'PercentageOutage']
        sum_max_at_below_5_date = group.loc[below_5_index, 'sum_max']

        minute_difference = (below_5_date - max_date).total_seconds() / 60
        hour_difference = minute_difference / 60
        day_difference = hour_difference / 24

        return pd.Series({
            'recovery_time_minutes': minute_difference,
            'recovery_time_hours': hour_difference,
            'recovery_time_days': day_difference,
            'max_date': max_date.strftime('%m/%d/%Y %H:%M'),
            'max_percent_outage': max_percentage_outage,
            'sum_max_at_max_date': sum_max_at_max_date,
            'below_5_date': below_5_date.strftime('%m/%d/%Y %H:%M'),
            'below_5_percent_outage': below_5_percentage_outage,
            'sum_max_at_below_5_date': sum_max_at_below_5_date
        })
    else:
        return pd.Series({
            'recovery_time_minutes': None,
            'recovery_time_hours': None,
            'recovery_time_days': None,
            'max_date': max_date.strftime('%m/%d/%Y %H:%M'),
            'max_percent_outage': max_percentage_outage,
            'sum_max_at_max_date': sum_max_at_max_date,
            'below_5_date': None,
            'below_5_percent_outage': None,
            'sum_max_at_below_5_date': None
        })

# -------------------------------
# Step 7: Group by key columns and apply recovery time calculation
# -------------------------------
result = df.groupby(['fips_code', 'county', 'state', 'total22_customers']).apply(calculate_recovery_times).reset_index()

# -------------------------------
# Step 8: Convert 'max_date' back to datetime for filtering
# -------------------------------
result['max_date'] = pd.to_datetime(result['max_date'], format='%m/%d/%Y %H:%M', errors='coerce')

# -------------------------------
# Step 9: Filter out rows based on cutoff date for max outage
# -------------------------------
cutoff_date = pd.to_datetime('2024-10-7')
result_filtered = result[result['max_date'] < cutoff_date]

# -------------------------------
# Step 10: Convert all date columns to MM/DD/YYYY HH:MM string format
# -------------------------------
date_columns = ['max_date', 'below_5_date']
for col in date_columns:
    result_filtered.loc[:, col] = pd.to_datetime(result_filtered[col], errors='coerce').dt.strftime('%m/%d/%Y %H:%M')

# -------------------------------
# Step 11: Print the filtered results
# -------------------------------
print(result_filtered)

# -------------------------------
# Step 12: Save to CSV
# -------------------------------
csv_output_file = 'Helene22x_RecoveryTimes.csv'
result_filtered.to_csv(csv_output_file, index=False)
print(f"Filtered results saved to CSV: {csv_output_file}")









# MAPPING HURRICANE AND OUTAGE INFORMATION FOR THE AFFECTED STATES



# -------------------------------
# Step 1: Import required libraries
# -------------------------------
import geopandas as gpd
import pandas as pd
import libpysal as ps
import networkx as nx

# -------------------------------
# Step 2: Load and preprocess data
# -------------------------------
# 2.1: Load the county shapefile and reproject to EPSG:5070
counties = gpd.read_file('C:/Users/aas0041/Documents/ArcGIS/Projects/National structures/tl_rd22_us_county.shp')
counties = counties.to_crs(epsg=5070)

# 2.2: Load the CSV data
data = pd.read_csv('C:/Users/aas0041/Desktop/eaglei_outages/Helene22x_RecoveryTimes.csv')

# ---- RENAME columns before merging ----
data.rename(columns={'fips_code': 'FIPS', 'state': 'StateName'}, inplace=True)

# Specify FIPS codes and StateNames to exclude
exclude_fips = [48301]
exclude_states = ["Connecticut"]

# Remove rows with these FIPS or StateNames
data = data[
    ~data['FIPS'].isin(exclude_fips) &
    ~data['StateName'].isin(exclude_states)
]

# 2.3: Merge shapefile and tabular data on FIPS
counties = counties.merge(data, on="FIPS")

# -------------------------------
# Step 3: Apply county-level filters for analysis
# -------------------------------
counties_subset = counties[
    (counties['max_percent_outage'] >= 5) &
    (counties['max_percent_outage'] <= 100) &
    (counties['total22_customers'] >= 1) &
    (counties['recovery_time_minutes'] >= 60)
].copy()

print(f" Selected counties after filtering: {counties_subset.shape[0]}")

# -------------------------------
# Step 4: Build spatial weight matrix (Queen contiguity)
# -------------------------------
w_subset = ps.weights.Queen.from_dataframe(counties_subset, ids='FIPS')
w_subset.transform = 'r'

# -------------------------------
# Step 5: Analyze spatial graph structure and connected components
# -------------------------------
# 5.1: Create graph of counties based on spatial neighbors
G_subset = nx.Graph()
for fips, neighbors in w_subset.neighbors.items():
    for neighbor in neighbors:
        G_subset.add_edge(fips, neighbor)

# 5.2: Find connected components (groups of spatially-connected counties)
connected_components_subset = list(nx.connected_components(G_subset))

# 5.3: Assign component ID to each county
component_map_subset = {node: idx for idx, component in enumerate(connected_components_subset) for node in component}
counties_subset['component'] = counties_subset['FIPS'].map(component_map_subset)


# -------------------------------
# Step 6: Map visualization with folium
# -------------------------------
import folium
from folium import LayerControl

# 6.1: Reproject counties to EPSG:4326 (latitude/longitude)
counties_subset = counties_subset.to_crs(epsg=4326)

# 6.2: Load and reproject state boundaries to match
states = gpd.read_file(
    'C:/Users/aas0041/Downloads/tl_2023_us_state/tl_2023_us_state.shp'
).to_crs(counties_subset.crs)

# 6.3: Create the interactive map showing county components
interactive_map_subset = counties_subset.explore(
    column="component",
    tooltip=["recovery_time_minutes", "max_percent_outage", "total22_customers", "StateName", "county"],
    popup=["FIPS", "county", "StateName"],
    tiles="CartoDB positron"
)

# 6.4: Overlay bold state boundaries
states.explore(
    m=interactive_map_subset,
    color="black",
    linewidth=0.3,
    fill=False,
    name="State Boundaries",
    tooltip="STUSPS"
)

# 6.5: Add state abbreviation labels (disabled by default)
show_labels = False
states_latlon = states.to_crs(epsg=4326)
states_latlon['label_point'] = states_latlon.representative_point()
if show_labels:
    for _, row in states_latlon.iterrows():
        folium.Marker(
            location=[row['label_point'].y, row['label_point'].x],
            icon=folium.DivIcon(
                html=f"""
                <div style="font-size: 12px; font-weight: bold; color: black;">
                    {row['STUSPS']}
                </div>
            """
        )
    ).add_to(interactive_map_subset)

# 6.6: Load hurricane path and reproject to EPSG:4326
hurricane_path = gpd.read_file('C:/Users/aas0041/Downloads/Hurricanes Path/AL092024_lin.shp').to_crs(epsg=4326)

# 6.7: Add the hurricane path to the interactive map
folium.GeoJson(
    hurricane_path,
    name="Hurricane Path",
    style_function=lambda x: {
        'color': 'blue',
        'weight': 4,
        'opacity': 0.7
    },
    tooltip=folium.GeoJsonTooltip(fields=["STORMTYPE", "SS"])
).add_to(interactive_map_subset)

# 6.8: Add a layer control to toggle layers
interactive_map_subset.add_child(LayerControl())

# 6.9: Save the interactive map as HTML
interactive_map_subset.save("d22x_Helene_component_map.html")
print("Interactive map has been saved as 'd22x_Helene_component_map.html'")
