



                        # SVI CALCULATION

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from factor_analyzer import Rotator


# -------------------------------
# Step 1: Define output file path and helper function
# -------------------------------

output_file_path = "SVI Result.txt"
output_file = open(output_file_path, "w", encoding="utf-8")

def print_and_save(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)

# -------------------------------
# Step 2: Read and preprocess input data
# -------------------------------

file_path = 'C:/Users/aas0041/Desktop/OutageData/AllSvi00test.csv'
df = pd.read_csv(file_path, nrows=100)

# Classify columns according to themes
theme_1_columns = ['Over 65 yrs', 'Under 5 yrs', 'At risk percentage', 'Nurs Hom', 'Amblry diffclty', 
                   'CHD_CrudePrev', 'DIABETES_CrudePrev', 'CASTHMA_CrudePrev', 'Heart D death', 'Diabetes death', 'Asthma death']

theme_2_columns = ['Median age ', 'Child under 6 yrs', 
                   'HH size', 'Median income', 'Povrty lvl', 'No diploma', 'Lvng alone', '2unit or more']

theme_3_columns = ['Child under 18yrs', 'HH size', 'Cattle per 1000', 
                   'Hogs per 1000', 'Sheep per 1000', 'Poultry per 1000', 'Goats per 1000', 'Educ job', 'Health job', 'Service job', 
                   'Natural rescs', 'Production job', 'No vhcl', 'Disability', 'Median income', 'Povrty lvl']

# Columns to calculate complements for themes
theme_1_complement_columns = []
theme_2_complement_columns = ['Median age ', 'Median income']
theme_3_complement_columns = [
    'Cattle per 1000', 'Hogs per 1000', 'Sheep per 1000', 'Poultry per 1000', 'Goats per 1000',
    'Educ job', 'Health job', 'Service job', 'Natural rescs', 'Production job', 'Median income'
]
county_column = ['Geographic Area']

# -------------------------------
# Step 3: Define utility functions
# -------------------------------

def normalize(df, columns):
    df[columns] = df[columns].astype(float)
    scaler = MinMaxScaler()
    df.loc[:, columns] = scaler.fit_transform(df.loc[:, columns])
    return df

def apply_complements(df, complement_columns):
    if complement_columns:
        df.loc[:, complement_columns] = 1 - df.loc[:, complement_columns]
    return df

def perform_pca(df, columns, theme_name):
    pca = PCA() 
    principal_components = pca.fit_transform(df[columns])
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
    explained_variance_ratio = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    loadings = pca.components_.T * np.sqrt(eigenvalues)
    loadings_df = pd.DataFrame(loadings, index=columns, columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
    
    # Select components that explain at least 85% variability
    cumulative_variance = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_variance >= 0.85) + 1
    selected_variance_ratio = explained_variance_ratio[:num_components]
    selected_pca_df = pca_df.iloc[:, :num_components]
    
    # Calculate total variance explained by selected components
    total_variance_selected = np.sum(selected_variance_ratio)
    total_variance_percentage = total_variance_selected * 100
    return selected_pca_df, explained_variance_ratio, eigenvalues, loadings_df, total_variance_percentage

def apply_varimax_with_kaiser_normalization(loadings_df):
    kaiser_normalized = loadings_df.div(np.sqrt((loadings_df**2).sum(axis=0)), axis=1)
    rotator = Rotator(method='varimax')
    rotated_loadings = rotator.fit_transform(kaiser_normalized)
    scaling_factors = np.sqrt((loadings_df**2).sum(axis=0)).values
    rotated_loadings = rotated_loadings * scaling_factors[:, np.newaxis]
    rotated_loadings_df = pd.DataFrame(rotated_loadings, index=loadings_df.index, columns=loadings_df.columns)
    return rotated_loadings_df

def calculate_l2_norm(pca_df):
    l2_norms = np.linalg.norm(pca_df.values, axis=1)
    return l2_norms

def pareto_ranking(norms_df, columns):
    ranks = {}
    rank = 1
    remaining_counties = norms_df[['Geographic Area'] + columns].copy()
    while not remaining_counties.empty:
        pareto_front = []
        for i, row in remaining_counties.iterrows():
            county_dominated = False
            for j, other_row in remaining_counties.iterrows():
                if i != j:
                    if all(other_row[col] >= row[col] for col in columns) and any(other_row[col] > row[col] for col in columns):
                        county_dominated = True
                        break
            if not county_dominated:
                pareto_front.append(row)
        for row in pareto_front:
            ranks[row['Geographic Area']] = (rank, *row[columns].values)
            remaining_counties = remaining_counties[remaining_counties['Geographic Area'] != row['Geographic Area']]
        rank += 1
    return ranks

# -------------------------------
# Step 4: Create and normalize separate DataFrames for each theme
# -------------------------------

theme_1_df = df[county_column + theme_1_columns].copy()
theme_2_df = df[county_column + theme_2_columns].copy()
theme_3_df = df[county_column + theme_3_columns].copy()

theme_1_df = normalize(theme_1_df, theme_1_columns)
theme_1_df = apply_complements(theme_1_df, theme_1_complement_columns)

theme_2_df = normalize(theme_2_df, theme_2_columns)
theme_2_df = apply_complements(theme_2_df, theme_2_complement_columns)

theme_3_df = normalize(theme_3_df, theme_3_columns)
theme_3_df = apply_complements(theme_3_df, theme_3_complement_columns)

# -------------------------------
# Step 5: Perform PCA for each theme
# -------------------------------

theme_1_pca_df, theme_1_explained_variance, theme_1_eigenvalues, theme_1_loadings, theme_1_var_percent = perform_pca(theme_1_df, theme_1_columns, "Theme 1")
theme_2_pca_df, theme_2_explained_variance, theme_2_eigenvalues, theme_2_loadings, theme_2_var_percent = perform_pca(theme_2_df, theme_2_columns, "Theme 2")
theme_3_pca_df, theme_3_explained_variance, theme_3_eigenvalues, theme_3_loadings, theme_3_var_percent = perform_pca(theme_3_df, theme_3_columns, "Theme 3")

# -------------------------------
# Step 6: Apply Varimax rotation with Kaiser normalization
# -------------------------------

theme_1_rotated_loadings = apply_varimax_with_kaiser_normalization(theme_1_loadings)
theme_2_rotated_loadings = apply_varimax_with_kaiser_normalization(theme_2_loadings)
theme_3_rotated_loadings = apply_varimax_with_kaiser_normalization(theme_3_loadings)

# -------------------------------
# Step 7: Calculate L2 norms for each theme's principal components
# -------------------------------

theme_1_l2_norms = calculate_l2_norm(theme_1_pca_df)
theme_2_l2_norms = calculate_l2_norm(theme_2_pca_df)
theme_3_l2_norms = calculate_l2_norm(theme_3_pca_df)

# -------------------------------
# Step 8: Combine L2 norms into a single DataFrame
# -------------------------------

l2_norms_df = pd.DataFrame({
    'Geographic Area': df['Geographic Area'],
    'Theme_1_L2': theme_1_l2_norms,
    'Theme_2_L2': theme_2_l2_norms,
    'Theme_3_L2': theme_3_l2_norms
})

# -------------------------------
# Step 9: Perform Pareto ranking with the dominance criteria
# -------------------------------

pareto_ranks_all = pareto_ranking(l2_norms_df, ['Theme_1_L2', 'Theme_2_L2', 'Theme_3_L2'])
pareto_ranks_1_2 = pareto_ranking(l2_norms_df, ['Theme_1_L2', 'Theme_2_L2'])
pareto_ranks_1_3 = pareto_ranking(l2_norms_df, ['Theme_1_L2', 'Theme_3_L2'])
pareto_ranks_2_3 = pareto_ranking(l2_norms_df, ['Theme_2_L2', 'Theme_3_L2'])
pareto_ranks_1 = pareto_ranking(l2_norms_df, ['Theme_1_L2'])
pareto_ranks_2 = pareto_ranking(l2_norms_df, ['Theme_2_L2'])
pareto_ranks_3 = pareto_ranking(l2_norms_df, ['Theme_3_L2'])

# Convert the ranks dictionaries to DataFrames for better readability
pareto_ranks_dfs = {}
pareto_ranks_dfs['All Themes'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_all.items()],
                                              columns=['Geographic Area', 'Rank', 'Theme_1_L2', 'Theme_2_L2', 'Theme_3_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 1 and 2'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_1_2.items()],
                                                 columns=['Geographic Area', 'Rank', 'Theme_1_L2', 'Theme_2_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 1 and 3'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_1_3.items()],
                                                 columns=['Geographic Area', 'Rank', 'Theme_1_L2', 'Theme_3_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 2 and 3'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_2_3.items()],
                                                 columns=['Geographic Area', 'Rank', 'Theme_2_L2', 'Theme_3_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 1'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_1.items()],
                                           columns=['Geographic Area', 'Rank', 'Theme_1_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 2'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_2.items()],
                                           columns=['Geographic Area', 'Rank', 'Theme_2_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 3'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_3.items()],
                                           columns=['Geographic Area', 'Rank', 'Theme_3_L2']).sort_values(by='Rank')

# -------------------------------
# Step 10: Normalize and reverse the ranks
# -------------------------------

scaler = MinMaxScaler()
for theme_key, df_ in pareto_ranks_dfs.items():
    df_['Reversed Rank'] = df_['Rank'].max() + 1 - df_['Rank']
    df_['Normalized Reversed Rank'] = scaler.fit_transform(df_[['Reversed Rank']])
    pareto_ranks_dfs[theme_key] = df_

# -------------------------------
# Step 11: Print and save Pareto ranking results
# -------------------------------

for key, df_ in pareto_ranks_dfs.items():
    print_and_save(f"\n================ Results for {key} Pareto Ranking ================\n")
    print_and_save(df_)
    print_and_save('\n')

# -------------------------------
# Step 12: Save all Pareto rank DataFrames to an Excel file
# -------------------------------

output_excel_path = 'Allnormalized_pareto_ranks_all_2024themes.xlsx'
with pd.ExcelWriter(output_excel_path) as writer:
    for theme_key, df_ in pareto_ranks_dfs.items():
        df_[['Geographic Area', 'Rank', 'Reversed Rank', 'Normalized Reversed Rank']].to_excel(writer, sheet_name=theme_key, index=False)

# -------------------------------
# Step 13: Print and save  PCA results for each theme
# -------------------------------

print_and_save("\n================ Results for Theme 1 PCA ====================\n")
print_and_save("Selected PCA DataFrame:\n", theme_1_pca_df.head())
print_and_save()
print_and_save("Selected Explained Variance Ratio:", theme_1_explained_variance[:len(theme_1_pca_df.columns)])
print_and_save()
print_and_save(f"Selected Components explain {theme_1_var_percent:.2f}% of variance.")
print_and_save()
print_and_save("L2 Norms of Selected Principal Components:\n", theme_1_l2_norms)
print_and_save('\n')

print_and_save("\n================ Results for Theme 2 PCA ====================\n")
print_and_save("Selected PCA DataFrame:\n", theme_2_pca_df.head())
print_and_save()
print_and_save("Selected Explained Variance Ratio:", theme_2_explained_variance[:len(theme_2_pca_df.columns)])
print_and_save()
print_and_save(f"Selected Components explain {theme_2_var_percent:.2f}% of variance.")
print_and_save()
print_and_save("L2 Norms of Selected Principal Components:\n", theme_2_l2_norms)
print_and_save('\n')

print_and_save("\n================ Results for Theme 3 PCA ====================\n")
print_and_save("Selected PCA DataFrame:\n", theme_3_pca_df.head())
print_and_save()
print_and_save("Selected Explained Variance Ratio:", theme_3_explained_variance[:len(theme_3_pca_df.columns)])
print_and_save()
print_and_save(f"Selected Components explain {theme_3_var_percent:.2f}% of variance.")
print_and_save()
print_and_save("L2 Norms of Selected Principal Components:\n", theme_3_l2_norms)
print_and_save('\n')

print_and_save("\n================ Combined L2 Norms of All Themes ====================\n")
print_and_save(l2_norms_df)

print(f"SVI results have been saved to {output_file_path}")
output_file.close()












                        # SVI PLOT

# -------------------------------
# Step 1: Load the county shapefile
# -------------------------------
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

shapefile_path = 'C:/Users/aas0041/Documents/ArcGIS/Projects/National structures/tl_rd22_us_county.shp'
counties = gpd.read_file(shapefile_path)

# Ensure GEOID is a string for consistency
counties['GEOID'] = counties['GEOID'].astype(str)

# -------------------------------
# Step 2: Load the CSV file with SVI data
# -------------------------------
csv_path = 'C:/Users/aas0041/Desktop/OutageData/AllParetoWithFIPS.csv'
data = pd.read_csv(csv_path)

# Ensure GEOID in the CSV is also a string and pad leading zeros to match shapefile
data['GEOID'] = data['GEOID'].astype(str).str.zfill(len(counties['GEOID'].iloc[0]))

# -------------------------------
# Step 3: Filter shapefile to include only counties present in the CSV file
# -------------------------------
filtered_counties = counties[counties['GEOID'].isin(data['GEOID'])]

# -------------------------------
# Step 4: Merge the filtered shapefile with the CSV data
# -------------------------------
merged = filtered_counties.merge(data, on="GEOID", how="left")

# -------------------------------
# Step 5: Log transformation of the All_SVI column
# -------------------------------
merged['Log_All_SVI'] = np.log1p(merged['All_SVI'])  # Log transformation of SVI values

# -------------------------------
# Step 6: Bin SVI values into categories and calculate value ranges
# -------------------------------
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
# Equal-frequency binning using quantiles
merged['SVI_Category'] = pd.qcut(merged['Log_All_SVI'], q=5, labels=labels)

# Calculate actual value ranges for each SVI category (back-transform)
actual_ranges = []
for i in range(len(labels)):
    category_values = merged[merged['SVI_Category'] == labels[i]]['Log_All_SVI']
    actual_min = category_values.min()
    actual_max = category_values.max()
    actual_ranges.append(f"{np.expm1(actual_min):.2f} - {np.expm1(actual_max):.2f}") # Transform back to original scale

# -------------------------------
# Step 7: Load and filter the state boundary shapefile
# -------------------------------
state_shapefile_path = 'C:/Users/aas0041/Downloads/tl_2023_us_state/tl_2023_us_state.shp'
states = gpd.read_file(state_shapefile_path)

# Filter states based on counties' STATEFP codes
relevant_states = filtered_counties['STATEFP'].unique()
filtered_states = states[states['STATEFP'].isin(relevant_states)]

# Make states in the same CRS as counties
filtered_states = filtered_states.to_crs(counties.crs)


# -------------------------------
# Step 8: Plot the SVI categories and state boundaries
# -------------------------------
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot only the filtered counties with SVI categories
merged.plot(
    column='SVI_Category', 
    cmap='YlOrRd', 
    linewidth=0.8, 
    edgecolor='0.8', 
    ax=ax, 
    legend=False
)

# Plot the filtered state boundaries
filtered_states.boundary.plot(ax=ax, color='black', linewidth=1)


# ---- Add state abbreviations at an interior point ----
for _, row in filtered_states.iterrows():
    x, y = row.geometry.representative_point().coords[0]
    ax.text(
        x, y, row['STUSPS'],
        ha='center', va='center', fontsize=9, fontweight='bold', color='blue',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]  # white halo for readability
    )

# Create custom legend with categories and actual SVI ranges
handles = [
    plt.Line2D(
        [0], [0], 
        marker='o', 
        color='w', 
        label=f"{labels[i]} ({actual_ranges[i]})",  # Append actual range to the category label
        markersize=10, 
        markerfacecolor=plt.cm.YlOrRd((i + 0.5) / len(labels))
    ) 
    for i in range(len(labels))
]

# Add the custom legend
legend = ax.legend(
    handles=handles, 
    title="SVI Categories (with Ranges)", 
    loc='center', 
    bbox_to_anchor=(0.25, 0.8),  # Center of the map
    fontsize=10, 
    title_fontsize=12
)

# Add title and remove axes
column_to_plot = 'All_SVI'
ax.set_title(f"Categorized Spatial Distribution of {column_to_plot}", fontsize=15)
ax.axis('off')

# -------------------------------
# Step 9: Show and save the plot
# -------------------------------
plt.show()

output_path = "SVI Map.png"
fig.savefig(output_path, dpi=300, bbox_inches='tight')
