


# Eagle I

# -------------------------------
# Step 0: Import libraries
# -------------------------------
import geopandas as gpd
import pandas as pd
import libpysal as ps
from spreg import ML_Lag, ML_Error
import numpy as np
from scipy.sparse import block_diag
from esda.moran import Moran
from sklearn.preprocessing import OneHotEncoder
from libpysal.weights import W, WSP
import sys

# -------------------------------
# Step 1: Define output file path and helper function
# -------------------------------
output_file_path = "Spatial Regression Result.txt"
output_file = open(output_file_path, "w")
def print_and_save(*args, **kwargs):
    """Prints to console and writes to a file."""
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)

# -------------------------------
# Step 2: Load and preprocess the shapefile
# -------------------------------
counties = gpd.read_file('C:/Users/aas0041/Documents/ArcGIS/Projects/National structures/tl_rd22_us_county.shp')
counties = counties.to_crs(epsg=5070)  # Reproject to a projected CRS

# -------------------------------
# Step 3: Load and merge Excel data
# -------------------------------
data = pd.read_excel('C:/Users/aas0041/Desktop/eaglei_outages/02_2022_Combined_Hurricane_Data_Normalized.xlsx')
counties['FIPS'] = counties['FIPS'].astype(str)
data['FIPS'] = data['FIPS'].astype(str)
counties = counties.merge(data, on="FIPS")
print(f"Total number of counties before filtering: {len(counties)}")

# -------------------------------
# Step 4: Create a unique identifier
# -------------------------------
counties['unique_id'] = (counties['FIPS'].astype(str) + '_' +
                           counties['Hurricane'].astype(str) + '_' +
                           counties['Year'].astype(str))

# -------------------------------
# Step 5: Define selection criteria and subset data
# -------------------------------
selected_years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
selected_hurricanes = ['Delta', 'Dorian', 'Florence', 'Hanna', 'Harvey', 'Ian', 'Ida', 'Idalia', 
                       'Irma', 'Isaias', 'Laura', 'Michael', 'Nicholas', 'Sally', 'Zeta', 'Nicole',
                       'Beryl', 'Debby', 'Francine', 'Helene', 'Milton']
counties_subset = counties[
    (counties['max_percent_outage'] >= 5) &
    (counties['max_percent_outage'] <= 100) &
    (counties['total22_customers'] >= 1) &
    (counties['recovery_time_minutes'] >= 60) &
    (counties['Year'].isin(selected_years)) &
    (counties['Hurricane'].isin(selected_hurricanes))
].copy()
print_and_save(f"Total number of counties after filtering: {len(counties_subset)}")

# -------------------------------
# Step 6: Identify and Print Duplicate Counties
# -------------------------------
duplicates = counties_subset[counties_subset.duplicated(subset=['unique_id'], keep=False)]
if not duplicates.empty:
    print_and_save("Duplicate counties found before resolving:")
    print_and_save(duplicates[['FIPS', 'state', 'county', 'Hurricane', 'Year', 'unique_id']])
else:
    print("No duplicate counties found.")

# -------------------------------
# Step 7: Resolve duplicate unique IDs
# -------------------------------
if counties_subset['unique_id'].duplicated().sum() > 0:
    counties_subset['unique_id'] = (
        counties_subset.groupby('unique_id').cumcount().astype(str) + '_' + counties_subset['unique_id']
    )

# -------------------------------
# Step 8: Sort counties_subset explicitly by Hurricane and unique_id
# -------------------------------
counties_subset = counties_subset.sort_values(['Hurricane', 'unique_id']).reset_index(drop=True)

# -------------------------------
# Step 9: Build concatenated DataFrame by hurricane (for block-diagonal construction)
# -------------------------------
ordered_hurricanes = list(counties_subset['Hurricane'].unique())
concat = pd.concat(
    [counties_subset[counties_subset['Hurricane'] == h] for h in ordered_hurricanes],
    axis=0
).reset_index(drop=True)

# -------------------------------
# Step 10: Check row order alignment between main data and block-diagonal
# -------------------------------
if not concat['unique_id'].equals(counties_subset['unique_id']):
    mismatch = (concat['unique_id'] != counties_subset['unique_id'])
    print_and_save("ERROR: The row order of counties in block-diagonal does NOT match the full data.")
    print_and_save(f"First 10 mismatches:\n{pd.DataFrame({'block_diag': concat['unique_id'], 'full': counties_subset['unique_id']})[mismatch].head(10)}")
else:
    print("Block-diagonal spatial weights will align properly with the main data order.")

# -------------------------------
# Step 11: Generate hurricane-specific Queen contiguity weights
# -------------------------------
hurricane_weights = {}
success = True

for hurricane in counties_subset['Hurricane'].unique():
    try:
        subset = counties_subset[counties_subset['Hurricane'] == hurricane]
        w = ps.weights.Queen.from_dataframe(subset, ids='unique_id')
        w.transform = 'r'
        hurricane_weights[hurricane] = w
    except Exception as e:
        print_and_save(f"Error creating spatial weights for hurricane {hurricane}: {e}")
        success = False

if success:
    print("Hurricane-Specific Spatial Weights successfully created for all hurricanes.")
else:
    print("Spatial Weights creation FAILED for one or more hurricanes.")

    
print_and_save("Hurricane-Specific Spatial Weights Details:")
for hurricane, weight_matrix in hurricane_weights.items():
    n = len(weight_matrix.neighbors)
    
    print_and_save(f"\nHurricane: {hurricane}")
    print_and_save(f"  Number of counties: {n}")

# -------------------------------
# Step 12: Create a regime map and align weights
# -------------------------------
regime_map = {h: idx for idx, h in enumerate(hurricane_weights.keys())}
counties_subset['regime_id'] = counties_subset['Hurricane'].map(regime_map)
regimes = counties_subset['regime_id'].values
weights_list = [hurricane_weights[h] for h in counties_subset['Hurricane'].unique()]

# -------------------------------
# Step 13: Create block-diagonal spatial weights matrix
# -------------------------------
try:
    block_diag_sparse = block_diag([w.sparse for w in weights_list], format="csr")
    print(f"Block-diagonal sparse matrix created successfully: Shape = {block_diag_sparse.shape}")

    W_dense = block_diag_sparse.toarray()
    print(f"Dense matrix created successfully: Shape = {W_dense.shape}")

    try:
        neighbors = {i: list(np.where(W_dense[i] > 0)[0]) for i in range(W_dense.shape[0])}
        weights = {i: [W_dense[i, j] for j in neighbors[i]] for i in neighbors}
        w_combined = W(neighbors, weights)
        w_combined.transform = 'r'
        print("Combined weights matrix successfully created as W.")
    except Exception as e:
        print(f"Error creating W combined weights: {e}")

except Exception as e:
    print(f"Error creating combined weights: {e}")

# -------------------------------
# Step 14: Define an elementwise custom log transformation function
# -------------------------------
def custom_log_transform_elementwise(x):
    result = np.empty_like(x, dtype=float)
    for idx, val in np.ndenumerate(x):
        if val == 0:
            print_and_save(f"Debug: Found zero at index {idx}, value: {val}")
            result[idx] = np.log(val + 1)
        else:
            result[idx] = np.log(val)
    return result

# -------------------------------
# Step 15: Add fixed effects using OneHotEncoder
# -------------------------------
encoder = OneHotEncoder(drop='first', sparse_output=False)
fixed_effects = pd.DataFrame(
    encoder.fit_transform(counties_subset[['Hurricane', 'state', 'County_class', 'Hurricane_risk', 'Wind_swath']]),
    columns=encoder.get_feature_names_out(['Hurricane', 'state', 'County_class', 'Hurricane_risk', 'Wind_swath']) 
)

# -------------------------------
# Step 16: Apply Corrected Log Transformation Elementwise
# -------------------------------
variables_to_log = ['max_percent_outage', 'GDP_c',
                      'Median income', 'Critical_facilities_Count_c', 'recovery_time_minutes']

# -------------------------------
# Step 17: Check for negative values before log transform
# -------------------------------
for var in variables_to_log:
    negative_count = (counties_subset[var] < 0).sum()
    if negative_count > 0:
        print(f"Warning: Variable '{var}' has {negative_count} negative values; "
                       "log transform may result in NaNs or errors.") 

for var in variables_to_log:
    counties_subset[var] = custom_log_transform_elementwise(counties_subset[var].values)

# -------------------------------
# Step 18: Check for NaNs after log transformation
# -------------------------------
nan_report = {}
for var in variables_to_log:
    nan_count = counties_subset[var].isna().sum()
    if nan_count > 0:
        nan_report[var] = nan_count
if nan_report:
    print(f"NaNs detected after log transformation: {nan_report}")
else:
    print("No NaNs detected in log-transformed variables.")

# -------------------------------
# Step 19: Combine fixed effects with independent variables
# -------------------------------
independent_vars = ['max_percent_outage', 'Median income', 'Critical_facilities_Count_c', 'GDP_c', 'All_SVI'] 
X = counties_subset[independent_vars].values
X_combined = np.hstack([X, fixed_effects.values])
w_combined.transform = 'r'

# -------------------------------
# Step 20: Fit Spatial Durbin Model (SDM) (Simplified)
# -------------------------------
def fit_sdm(y, dependent_var_name):
    try:
        X_final = np.hstack([X_combined, W_dense @ X])
        column_names = independent_vars + list(fixed_effects.columns) + \
                       [f"W_{var}" for var in independent_vars]
        model = ML_Lag(
            y, X_final, w_combined, name_y=dependent_var_name,
            name_x=column_names, name_w='Combined Weights', name_ds='counties_subset'
        )
        print_and_save("\n==================== Spatial Durbin Model (SDM) ====================")
        print_and_save(f"SDM MODEL SUMMARY FOR {dependent_var_name}:")
        print_and_save(model.summary)
        residuals = model.u
        moran_residuals = Moran(residuals, w_combined)
        print_and_save(f"Moran's I for SDM residuals of {dependent_var_name}: {moran_residuals.I}")
        print_and_save(f"p-value: {moran_residuals.p_sim}, z-score: {moran_residuals.z_sim}")
        return model
    except Exception as e:
        print_and_save(f"Error fitting SDM model: {e}")
        return None

# -------------------------------
# Step 21: Fit Spatial Autoregressive Model (SAR)
# -------------------------------
def fit_sar(y, dependent_var_name):
    try:
        model = ML_Lag(
            y, X_combined, w_combined, name_y=dependent_var_name,
            name_x=list(independent_vars) + list(fixed_effects.columns),
            name_w='Combined Weights', name_ds='counties_subset'
        )
        print_and_save("\n==================== Spatial Autoregressive Model (SAR) ====================")
        print_and_save(f"SAR MODEL SUMMARY FOR {dependent_var_name}:")
        print_and_save(model.summary)
        residuals = model.u
        moran_residuals = Moran(residuals, w_combined)
        print_and_save(f"Moran's I for SAR residuals of {dependent_var_name}: {moran_residuals.I}")
        print_and_save(f"p-value: {moran_residuals.p_sim}, z-score: {moran_residuals.z_sim}")
        return model
    except Exception as e:
        print_and_save(f"Error fitting SAR model: {e}")
        return None

# -------------------------------
# Step 22: Fit Spatial Error Model (SEM)
# -------------------------------
def fit_sem(y, dependent_var_name):
    try:
        model = ML_Error(
            y, X_combined, w_combined, name_y=dependent_var_name,
            name_x=list(independent_vars) + list(fixed_effects.columns),
            name_w='Combined Weights', name_ds='counties_subset'
        )
        print_and_save("\n==================== Spatial Error Model (SEM) ====================")
        print_and_save(f"SEM MODEL SUMMARY FOR {dependent_var_name}:")
        print_and_save(model.summary)
        residuals = model.u
        moran_residuals = Moran(residuals, w_combined)
        print_and_save(f"Moran's I for SEM residuals of {dependent_var_name}: {moran_residuals.I}")
        print_and_save(f"p-value: {moran_residuals.p_sim}, z-score: {moran_residuals.z_sim}")
        return model
    except Exception as e:
        print_and_save(f"Error fitting SEM model: {e}")
        return None

# -------------------------------
# Step 23: Fit the models for the dependent variable 'recovery_time_minutes'
# -------------------------------
y_recovery_time_minutes = counties_subset['recovery_time_minutes'].values
model_sdm = fit_sdm(y_recovery_time_minutes, 'recovery_time_minutes')
model_sar = fit_sar(y_recovery_time_minutes, 'recovery_time_minutes')
model_sem = fit_sem(y_recovery_time_minutes, 'recovery_time_minutes')


# -------------------------------
# Step 24: Close output file after saving results
# -------------------------------
output_file.close()
print(f"Model outputs have been saved to {output_file_path}")
