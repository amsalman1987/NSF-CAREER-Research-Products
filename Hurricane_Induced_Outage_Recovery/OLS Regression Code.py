

# Lagrange

# -------------------------------
# Step 0: Import required libraries
# -------------------------------
import geopandas as gpd
import pandas as pd
import libpysal as ps
from spreg import OLS
from spreg.diagnostics_sp import LMtests
import numpy as np
import statsmodels.api as sm
from scipy.sparse import block_diag
from libpysal.weights import WSP
from sklearn.preprocessing import OneHotEncoder

# -------------------------------
# Step 1: Define output file path
# -------------------------------
output_file_path = "OLS Regression Result.txt"
output_file = open(output_file_path, "w", encoding="utf-8")

def print_and_save(*args, **kwargs):
    """Prints to console and writes to a file."""
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)

# -------------------------------
# Step 2: Define custom log transformation function
# -------------------------------
def custom_log_transform_elementwise(x):
    result = np.empty_like(x, dtype=float)
    for idx, val in np.ndenumerate(x):
        if val == 0:
            print(f"Debug: Found zero at index {idx}, value: {val}")
            result[idx] = np.log(val + 1)
        else:
            result[idx] = np.log(val)
    return result

# -------------------------------
# Step 3: Load and preprocess data
# -------------------------------

counties = gpd.read_file('C:/Users/aas0041/Documents/ArcGIS/Projects/National structures/tl_rd22_us_county.shp')
counties = counties.to_crs(epsg=5070)
data = pd.read_excel('C:/Users/aas0041/Desktop/eaglei_outages/02_2022_Combined_Hurricane_Data_Normalized.xlsx')
counties['FIPS'] = counties['FIPS'].astype(str)
data['FIPS'] = data['FIPS'].astype(str)
counties = counties.merge(data, on="FIPS")

counties['unique_id'] = counties['FIPS'].astype(str) + '_' + counties['Hurricane'].astype(str) + '_' + counties['Year'].astype(str)

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

# Resolve duplicate unique IDs if any
if counties_subset['unique_id'].duplicated().sum() > 0:
    counties_subset['unique_id'] = (
        counties_subset.groupby('unique_id').cumcount().astype(str) + '_' + counties_subset['unique_id']
    )

# Sort counties_subset explicitly by Hurricane
counties_subset = counties_subset.sort_values(['Hurricane', 'unique_id']).reset_index(drop=True)

# -------------------------------
# Step 4: Generate hurricane-specific spatial weights
# -------------------------------
hurricane_weights = {}
for hurricane in counties_subset['Hurricane'].unique():
    subset = counties_subset[counties_subset['Hurricane'] == hurricane]
    if len(subset) > 1:
        w = ps.weights.Queen.from_dataframe(subset, ids='unique_id')
        w.transform = 'r'
        hurricane_weights[hurricane] = w
    else:
        print(f" Skipping {hurricane}: Not enough counties for weight matrix.")

# -------------------------------
# Step 5: Build block-diagonal weights matrix
# -------------------------------
ordered_hurricanes = list(hurricane_weights.keys())
dfs = []
weights_list = []
for h in ordered_hurricanes:
    df_h = counties_subset[counties_subset['Hurricane'] == h].copy()
    df_h = df_h.sort_values('unique_id')
    dfs.append(df_h)
    weights_list.append(hurricane_weights[h].sparse)
full_data = pd.concat(dfs, ignore_index=True)

# -------------------------------
# Step 6: Apply log transformation to selected variables
# -------------------------------
log_transform_vars = ['max_percent_outage', 'GDP_c', 'Median income', 'Critical_facilities_Count_c', 'recovery_time_minutes']
for var in log_transform_vars:
    if var in full_data.columns:
        full_data[var] = custom_log_transform_elementwise(full_data[var].values)

# -------------------------------
# Step 7: Create combined weights object
# -------------------------------
W_block = block_diag(weights_list, format="csr")
print(f"Block-diagonal sparse matrix created: Shape = {W_block.shape}")
w_combined = WSP(W_block)
w_combined.transform = 'r'

# -------------------------------
# Step 8: Add fixed effects
# -------------------------------
encoder = OneHotEncoder(drop='first', sparse_output=False)
fixed_effects = pd.DataFrame(
    encoder.fit_transform(full_data[['Hurricane', 'state', 'County_class', 'Hurricane_risk', 'Wind_swath']]),
    columns=encoder.get_feature_names_out(['Hurricane', 'state', 'County_class', 'Hurricane_risk', 'Wind_swath'])
)
fixed_effects['unique_id'] = full_data['unique_id']
full_data = full_data.merge(fixed_effects, on='unique_id', how='left')

# -------------------------------
# Step 9: Prepare variables for modeling
# -------------------------------
independent_vars = ['max_percent_outage', 'Median income', 'Critical_facilities_Count_c', 'GDP_c', 'All_SVI']
X_base = full_data[independent_vars].values
fixed_cols = list(encoder.get_feature_names_out(['Hurricane', 'state', 'County_class', 'Hurricane_risk', 'Wind_swath']))
X_fixed = full_data[fixed_cols].values
X_combined = np.hstack([X_base, X_fixed])
X_combined = sm.add_constant(X_combined)
var_names = ["const"] + independent_vars + fixed_cols



y = full_data['recovery_time_minutes'].values


# -------------------------------
# Step 10: Fit OLS (spreg) and run Lagrange Multiplier tests
# -------------------------------
ols_model = OLS(y, X_combined, name_y="Recovery Time Minutes", name_x=var_names)
print_and_save("\n==================== OLS Model Summary (spreg) ====================")
print_and_save(ols_model.summary)

lm_results = LMtests(ols_model, w_combined)
print_and_save("\n==================== Lagrange Multiplier (LM) Test Results ====================")
print_and_save(f" Standard LM Lag Test Statistic: {lm_results.lml[0]:.4f}, p-value: {lm_results.lml[1]:.4f}")
print_and_save(f" Standard LM Error Test Statistic: {lm_results.lme[0]:.4f}, p-value: {lm_results.lme[1]:.4f}")
print_and_save("\n==================== Robust Lagrange Multiplier Test Results ====================")
print_and_save(f" Robust LM Lag Test Statistic: {lm_results.rlml[0]:.4f}, p-value: {lm_results.rlml[1]:.4f}")
print_and_save(f" Robust LM Error Test Statistic: {lm_results.rlme[0]:.4f}, p-value: {lm_results.rlme[1]:.4f}")

# -------------------------------
# Step 11: Close the output file
# -------------------------------
print("\n All processing complete. Output has been saved to the output file.")
output_file.close()
