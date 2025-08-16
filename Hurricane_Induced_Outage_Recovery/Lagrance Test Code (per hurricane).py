


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
# Step 0.5: Define output file and print-and-save helper
# -------------------------------
output_file_path = "Lagrance Multiplier Test Result (per hurricane).txt"
output_file = open(output_file_path, "w", encoding="utf-8")

def print_and_save(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)

# -------------------------------
# Step 1: Define helper functions
# -------------------------------
# Step 1.1: Elementwise custom log transformation
def custom_log_transform_elementwise(x):
    result = np.empty_like(x, dtype=float)
    for idx, val in np.ndenumerate(x):
        if val == 0:
            print_and_save(f"Debug: Found zero at index {idx}, value: {val}")
            result[idx] = np.log(val + 1)
        else:
            result[idx] = np.log(val)
    return result

# Step 1.2: LM test function
def run_lm_tests(y, X, w, hurricane_name):
    try:
        print(f"\n Running LM tests for Hurricane: {hurricane_name}")
        print_and_save(f" Number of counties: {y.shape[0]}")
        
        X = X.astype(float)
        y = y.astype(float)
        
        # Fit OLS and Run LM tests
        X = sm.add_constant(X, has_constant='add')
        var_names = ["const"] + [f"Var{i}" for i in range(1, X.shape[1])]
        ols_model = OLS(y, X, name_y="Recovery Time", name_x=var_names)
        lm_test_results = LMtests(ols_model, w)
        
        
        # Extract and print standard LM tests results:
        lm_lag_stat = lm_test_results.lml[0]
        lm_lag_pval = lm_test_results.lml[1]
        lm_error_stat = lm_test_results.lme[0]
        lm_error_pval = lm_test_results.lme[1]
        
        print_and_save(f"\n LM Test Results for Hurricane {hurricane_name}:")
        print_and_save(f" LM Lag Statistic: {lm_lag_stat:.4f}, p-value: {lm_lag_pval:.4f}")
        print_and_save(f" LM Error Statistic: {lm_error_stat:.4f}, p-value: {lm_error_pval:.4f}")
        
        # Extract and print robust LM tests results:
        rlml_stat = lm_test_results.rlml[0]
        rlml_pval = lm_test_results.rlml[1]
        rlme_stat = lm_test_results.rlme[0]
        rlme_pval = lm_test_results.rlme[1]
        
        print_and_save(f" Robust LM Lag Statistic: {rlml_stat:.4f}, p-value: {rlml_pval:.4f}")
        print_and_save(f" Robust LM Error Statistic: {rlme_stat:.4f}, p-value: {rlme_pval:.4f}")
    except Exception as e:
        print_and_save(f" Error running LM tests for Hurricane {hurricane_name}: {e}")

# -------------------------------
# Step 2: Load and preprocess spatial and tabular data
# -------------------------------
# Step 2.1: Load shapefile
counties = gpd.read_file('C:/Users/aas0041/Documents/ArcGIS/Projects/National structures/tl_rd22_us_county.shp')
counties = counties.to_crs(epsg=5070)
data = pd.read_excel('C:/Users/aas0041/Desktop/eaglei_outages/02_2022_Combined_Hurricane_Data_Normalized.xlsx')
counties['FIPS'] = counties['FIPS'].astype(str)
data['FIPS'] = data['FIPS'].astype(str)
counties = counties.merge(data, on="FIPS")

# Step 2.6: Create unique ID
counties['unique_id'] = counties['FIPS'].astype(str) + '_' + counties['Hurricane'].astype(str) + '_' + counties['Year'].astype(str)

# -------------------------------
# Step 3: Filter counties for analysis
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
print_and_save(f" Total number of counties after filtering: {len(counties_subset)}")

# Step 3.1: Resolve duplicate unique IDs if any
if counties_subset['unique_id'].duplicated().sum() > 0:
    counties_subset['unique_id'] = (
        counties_subset.groupby('unique_id').cumcount().astype(str) + '_' + counties_subset['unique_id']
    )
# Step 3.2: Sort for spatial order
counties_subset = counties_subset.sort_values(['Hurricane', 'unique_id']).reset_index(drop=True)

# -------------------------------
# Step 4: Create hurricane-specific Queen spatial weights
# -------------------------------
hurricane_weights = {}
for hurricane in counties_subset['Hurricane'].unique():
    subset = counties_subset[counties_subset['Hurricane'] == hurricane]
    if len(subset) > 1:
        w = ps.weights.Queen.from_dataframe(subset, ids='unique_id')
        w.transform = 'r'
        hurricane_weights[hurricane] = w
    else:
        print_and_save(f" Skipping {hurricane}: Not enough counties for weight matrix.")

# -------------------------------
# Step 5: Build block-diagonal spatial weights matrix
# -------------------------------
# Step 5.1: Order dataframes and weight matrices by hurricane
ordered_hurricanes = list(hurricane_weights.keys())
dfs = []
weights_list = []
for h in ordered_hurricanes:
    df_h = counties_subset[counties_subset['Hurricane'] == h].copy()
    df_h = df_h.sort_values('unique_id')
    dfs.append(df_h)
    weights_list.append(hurricane_weights[h].sparse)
    
# Step 5.2: Concatenate all event data for full sample
full_data = pd.concat(dfs, ignore_index=True)

# Step 5.3: Create block-diagonal sparse matrix
W_block = block_diag(weights_list, format="csr")
print(f"Block-diagonal sparse matrix created: Shape = {W_block.shape}")

# Step 5.4: Convert to PySAL weights object
w_combined = WSP(W_block)
w_combined.transform = 'r'

# -------------------------------
# Step 6: Run LM tests for each hurricane
# -------------------------------
print("\n Running LM tests for each hurricane with state fixed effects and log transformation...")
independent_vars = ['max_percent_outage', 'GDP_c', 'Critical_facilities_Count_c', 'Median income', 'All_SVI']
log_transform_vars_normal = ['max_percent_outage', 'GDP_c', 'Critical_facilities_Count_c', 'Median income', 'recovery_time_minutes']

for hurricane, w in hurricane_weights.items():
    
    # Step 6.1: Select subset for this hurricane and align to spatial order
    subset = counties_subset[counties_subset["Hurricane"] == hurricane].copy()
    subset = subset.set_index('unique_id')
    try:
        subset = subset.loc[w.id_order].reset_index()
    except KeyError as ke:
        print_and_save(f" KeyError when reordering for {hurricane}: {ke}")
        continue
    if len(subset) > 1:
        
        # Step 6.2: Check state variation
        unique_states = subset['state'].nunique()
        
        # Step 6.3: Log-transform required variables
        for var in log_transform_vars_normal:
            if var in subset.columns:
                if np.any(subset[var] < 0):
                    print_and_save(f" Negative values detected in {var} for hurricane {hurricane}.")
                subset[var] = custom_log_transform_elementwise(subset[var].values)
                
        # Step 6.4: Prepare X (with or without state fixed effects)
        if unique_states < 2:
            X_hurricane = subset[independent_vars].values  # No state fixed effects
        else:
            encoder_state = OneHotEncoder(drop='first', sparse_output=False)
            state_fixed = pd.DataFrame(
                encoder_state.fit_transform(subset[['state']]),
                columns=encoder_state.get_feature_names_out(['state'])
            )
            X_base = subset[independent_vars].values
            X_state = state_fixed.values
            X_hurricane = np.hstack([X_base, X_state])
            
        # Step 6.5: Run LM tests
        y_hurricane = subset["recovery_time_minutes"].values
        print_and_save(f"\n--- Hurricane: {hurricane} ---")
        run_lm_tests(y_hurricane, X_hurricane, w, hurricane)
    else:
        print(f" Skipping LM test for {hurricane} (Not enough observations).")

print("\n LM Tests Completed.")
output_file.close()
