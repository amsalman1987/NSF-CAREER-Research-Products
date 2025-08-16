


# MORANS I (Autocorrelation check)

# Power Outage and recovery time data 


# -------------------------------
# Step 1: Import required libraries
# -------------------------------
import geopandas as gpd
import pandas as pd
import libpysal as ps
from esda.moran import Moran

# -------------------------------
# Step 2: Load the county shapefile
# -------------------------------
counties_shapefile = 'C:/Users/aas0041/Documents/ArcGIS/Projects/National structures/tl_rd22_us_county.shp'
counties = gpd.read_file(counties_shapefile)

# -------------------------------
# Step 3: Define list of CSV files (one per hurricane)
# -------------------------------
csv_files = [
    'C:/Users/aas0041/Desktop/eaglei_outages/Delta22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Dorian22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Florence22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Hanna22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Harvey22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Ian22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Ida22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Idalia22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Irma22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Isaias22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Laura22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Michael22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Nicholas22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Nicole22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Sally22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Zeta22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Beryl22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Debby22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Milton22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Francine22x_RecoveryTimes.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Helene22x_RecoveryTimes.csv',
]

# -------------------------------
# Step 4: Define function to print and save
# -------------------------------
output_file_path = "Morans I Results.txt"
output_file = open(output_file_path, "w", encoding="utf-8")

def print_and_save(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)

# -------------------------------
# Step 5: Loop through each hurricane CSV, process, and compute Moran's I
# -------------------------------
results = []

for file in csv_files:
    
    # 5.1: Load CSV file
    data = pd.read_csv(file)
    
    # 5.2: Rename fips_code column to FIPS for merging
    data.rename(columns={'fips_code': 'FIPS'}, inplace=True)
    
    # 5.3: Merge with counties shapefile on FIPS
    merged_data = counties.merge(data, on="FIPS")
    
    # 5.4: Apply selection criteria for valid counties
    counties_subset = merged_data[
        (merged_data['max_percent_outage'] >= 5) & 
        (merged_data['max_percent_outage'] <= 100) & 
        (merged_data['total22_customers'] >= 1) &          
        (merged_data['recovery_time_minutes'] >= 60)
    ].copy()
    
    # 5.5: Print count after filtering
    total_after = counties_subset.shape[0]
    print(f"--- {file.split('/')[-1]} ---")
    print(f"Counties after filtering: {total_after}")

    # 5.6: Create Queen contiguity spatial weights
    w = ps.weights.Queen.from_dataframe(counties_subset, ids='FIPS')
    w.transform = 'r'

    # 5.7: Extract variables for Moran's I
    max_percent_outage = counties_subset['max_percent_outage'].values
    recovery_time_minutes = counties_subset['recovery_time_minutes'].values

    # 5.8: Compute Moran's I
    moran_max_percent_outage = Moran(max_percent_outage, w)
    moran_recovery_time_minutes = Moran(recovery_time_minutes, w)

    # 5.9: Store results
    results.append({
        'file': file.split('/')[-1],
        'hurricane_name': file.split('/')[-1].replace('22x_RecoveryTimes.csv', ''),
        'total_after_filtering': total_after,
        'Moran_I_max_percent_outage': moran_max_percent_outage.I,
        'p-value_max_percent_outage': moran_max_percent_outage.p_sim,
        'z-score_max_percent_outage': moran_max_percent_outage.z_sim,
        'Moran_I_recovery_time_minutes': moran_recovery_time_minutes.I,
        'p-value_recovery_time_minutes': moran_recovery_time_minutes.p_sim,
        'z-score_recovery_time_minutes': moran_recovery_time_minutes.z_sim
    })
    print("---------------------------------------------------------------")

# -------------------------------
# Step 6: Output Moran's I results for each hurricane
# -------------------------------
for result in results:
    print_and_save("===============================================================")
    print_and_save(f"Results for Hurricane {result['hurricane_name']}:")
    print_and_save(f"  Counties selected after filtering: {result['total_after_filtering']}")
    print_and_save(f"  Moran's I for max_percent_outage: {result['Moran_I_max_percent_outage']:.4f}")
    print_and_save(f"    p-value: {result['p-value_max_percent_outage']:.4f}")
    print_and_save(f"    z-score: {result['z-score_max_percent_outage']:.4f}")
    print_and_save(f"  Moran's I for recovery_time_minutes: {result['Moran_I_recovery_time_minutes']:.4f}")
    print_and_save(f"    p-value: {result['p-value_recovery_time_minutes']:.4f}")
    print_and_save(f"    z-score: {result['z-score_recovery_time_minutes']:.4f}")
    print_and_save("===============================================================")

output_file.close()
print(f"\nAll Moran's I results have been saved to {output_file_path}")








# PLot of Morans I values and thier significance level

# -------------------------------
# Step 1: Import required libraries
# -------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# -------------------------------
# Step 2: Convert Moran's I results to DataFrame and prepare labels
# -------------------------------
# Assume 'results' is already defined as a list of dicts or records from previous computations
results_df = pd.DataFrame(results)

# Extract hurricane names for the index/labels
results_df['Hurricane'] = results_df['hurricane_name']

# -------------------------------
# Step 3: Select only desired columns for the heatmap
# -------------------------------
heatmap_df = results_df.set_index('Hurricane')[
    ['Moran_I_max_percent_outage', 'Moran_I_recovery_time_minutes']
]
heatmap_df.columns = [
    "Max % Outage",
    "Recovery Time (min)"
]

# -------------------------------
# Step 4: Prepare mask for statistical significance (p < 0.05)
# -------------------------------
sig_level = 0.05
p_cols = [
    "p-value_max_percent_outage",
    "p-value_recovery_time_minutes"
]
sig_mask = np.zeros(heatmap_df.shape, dtype=bool)
for i, (col, p_col) in enumerate(zip(heatmap_df.columns, p_cols)):
    sig_mask[:, i] = results_df[p_col].values < sig_level

# -------------------------------
# Step 5: Prepare cell annotations (Moran's I rounded to two decimals)
# -------------------------------
annot = heatmap_df.applymap(lambda v: f"{v:.2f}")

# -------------------------------
# Step 6: Plot the heatmap and overlay blue ellipses for not-significant results
# -------------------------------
plt.figure(figsize=(8, 7))
ax = sns.heatmap(
    heatmap_df,
    annot=annot.values,
    fmt="",
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Moran's I"}
)
plt.title("Global Moran's I by Hurricane\n(Blue circle: Not significant, p ≥ 0.05)")
plt.ylabel("Hurricane")
plt.xlabel("Variable")

# Overlay blue ellipses on not-significant values
for i in range(heatmap_df.shape[0]):  # rows (hurricanes)
    for j in range(heatmap_df.shape[1]):  # columns (variables)
        if not sig_mask[i, j]:
            ellipse = Ellipse(
                (j+0.5, i+0.5), 0.7, 0.7,  # Centered on cell
                fill=False, edgecolor='blue', linewidth=2
            )
            ax.add_patch(ellipse)

plt.tight_layout()
plt.savefig("Morans I_Heatmap.png", dpi=300, bbox_inches='tight')
plt.show()






