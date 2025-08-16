

# EXTRACT THE STORM ID AND THE RADII FROM HURRICANE DATA CSV FILES

# -------------------------------
# Step 1: Import required libraries
# -------------------------------

import pandas as pd

# -------------------------------
# Step 2: Define file paths
# -------------------------------


# 2.1: Existing hurricane CSV file paths

csv_file_paths = [
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Deltacounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Doriancounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Florencecounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Hannacounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Harveycounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Iancounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Idacounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Idaliacounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Irmacounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Isaiascounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Lauracounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Michaelcounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Nicholascounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Nicolecounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Sallycounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Zetacounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Berylcounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Debbycounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Miltoncounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Francinecounties_epsg5070.csv',
    'C:/Users/aas0041/Desktop/eaglei_outages/Overlapped counties/overlapping_Helenecounties_epsg5070.csv',
]

# 2.2: Base file paths (Hurricane_RecoveryTimes.csv)

base_file_paths = [
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
# Step 3: Loop through file pairs and process
# -------------------------------
for csv_path, base_path in zip(csv_file_paths, base_file_paths):
    
    # --- 3.1: Load the current base file (CSV)
    base_df = pd.read_csv(base_path)
    
    # --- 3.2: Load the corresponding CSV file
    csv_df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    
    # --- 3.3: Rename GEOID to fips_code if present
    if 'GEOID' in csv_df.columns:
        csv_df.rename(columns={'GEOID': 'fips_code'}, inplace=True)
    
    # --- 3.4: Select only the required columns
    csv_df = csv_df[['fips_code', 'RADII', 'STORMID']]
    
    # --- 3.5: For each fips_code, keep only the row with the highest RADII
    csv_df = csv_df.loc[csv_df.groupby('fips_code')['RADII'].idxmax()]
    
    # --- 3.6: Merge the base file with the CSV data
    merged_df = pd.merge(base_df, csv_df, on='fips_code', how='left')
    
    # --- 3.7: Save the resulting DataFrame to a new file as CSV, replacing 'RecoveryTimes' with 'SwathData'
    output_path = base_path.replace('RecoveryTimes', 'SwathData')
    merged_df.to_csv(output_path, index=False)
    
    print(f"Processed and saved: {output_path}")






# TAKE THE SAVED SWATH FILES AND RESAVE THEM WITH NEW NAME ADDING THE HURRICANE YEAR

# -------------------------------
# Step 1: Import required libraries
# -------------------------------
import os
import pandas as pd

# -------------------------------
# Step 2: Loop through each SwathData file to rename with year
# -------------------------------
for base_path in base_file_paths:
    
    # --- 2.1: Build SwathData file path from base_path
    swath_file = base_path.replace('RecoveryTimes', 'SwathData')
    
    # --- 2.2: Load SwathData file
    df = pd.read_csv(swath_file)
    
    # --- 2.3: Extract year from STORMID (last 4 chars of first non-null value)
    stormid_sample = df['STORMID'].dropna().astype(str).iloc[0]
    year_code = stormid_sample[-4:]
    
    # --- 2.4: Build new filename with year appended
    dir_path, original_filename = os.path.split(swath_file)
    filename_no_ext = os.path.splitext(original_filename)[0]
    new_filename = f"{filename_no_ext}_{year_code}.csv"
    new_path = os.path.join(dir_path, new_filename)
    
    # --- 2.5: Delete destination file if it already exists
    if os.path.exists(new_path):
        os.remove(new_path)
    
    # --- 2.6: Rename the file (move with new name)
    os.rename(swath_file, new_path)
    print(f"Renamed: {swath_file} → {new_path}")







# COMBINE ALL THE "HURRICANE + OUTAGE FILES" INTO ONE BIG FILE WITH THEIR CORRESPONDING HURRICANE AND YEAR COLUMNS

# -------------------------------
# Step 1: Import required libraries
# -------------------------------
import os
import pandas as pd
import re

# -------------------------------
# Step 2: Define file directory and regex pattern
# -------------------------------
swath_dir = 'C:/Users/aas0041/Desktop/eaglei_outages/'
pattern = re.compile(r'([A-Za-z]+)22x_SwathData_(\d{4})\.csv')

# -------------------------------
# Step 3: Collect and process all SwathData files
# -------------------------------
all_data = []

for filename in os.listdir(swath_dir):
    # --- 3.1: Match file pattern
    match = pattern.match(filename)
    if match:
        hurricane = match.group(1)   # e.g., 'Michael'
        year = match.group(2)        # e.g., '2018'
        
        # --- 3.2: Load CSV and add columns
        full_path = os.path.join(swath_dir, filename)
        df = pd.read_csv(full_path)
        df['Hurricane'] = hurricane
        df['Year'] = int(year)
        all_data.append(df)

# -------------------------------
# Step 4: Concatenate and save combined DataFrame
# -------------------------------
combined_df = pd.concat(all_data, ignore_index=True)
output_file = os.path.join(swath_dir, '00_2022_Combined_Hurricane_Data.csv')
combined_df.to_csv(output_file, index=False)

print(f"Combined file saved: {output_file}")








# ADD INDEPENDENT VARIABLES TO THE RESULTING "HURRICANE + OUTAGE FILE" 


# -------------------------------
# Step 1: Import required libraries
# -------------------------------
import pandas as pd

# -------------------------------
# Step 2: Load data files
# -------------------------------
file1 = pd.read_csv('C:/Users/aas0041/Desktop/eaglei_outages/00_2022_Combined_Hurricane_Data.csv')
file2 = pd.read_csv('C:/Users/aas0041/Desktop/Journal Data/Complete_Socioeconomic_Data.csv')

# -------------------------------
# Step 3: Preprocess file1 columns for merging
# -------------------------------
file1.rename(columns={'fips_code': 'FIPS'}, inplace=True)

# -------------------------------
# Step 4: Select columns for merging
# -------------------------------
columns_to_merge = [
    'FIPS', 'POPESTIMATE2022', 'NAME', 'Geographic Area',
    'Establishment', 'Employment', 'Total Wages', 'GDP', 'Education_Count',
    'Law_Enforcement_Count', 'Medical_Emergency_Count', 'Median income', '2013 code',
    'HRCN_RISKS', 'HRCN_RISKR', 'CFLD_RISKS', 'CFLD_RISKR', 'All_SVI', 
    'HealthPrep_SVI', 'HealthEvac_SVI', 'PrepEvac_SVI', 'Health_SVI', 
    'Prep_SVI', 'Evac_SVI'
]

# -------------------------------
# Step 5: Merge data on 'FIPS'
# -------------------------------
merged_df = pd.merge(file1, file2[columns_to_merge], on='FIPS', how='left')

# -------------------------------
# Step 6: Exclude specific FIPS and StateNames With missing data
# -------------------------------
exclude_fips = [48301]
exclude_states = ["Connecticut"]

filtered_df = merged_df[
    (~merged_df['FIPS'].isin(exclude_fips)) & 
    (~merged_df['state'].isin(exclude_states))
].copy()

# -------------------------------
# Step 7: Replace NaN values with 0 in certain columns
# -------------------------------
columns_to_replace = [
    'Education_Count', 'Law_Enforcement_Count', 'Medical_Emergency_Count', 'HRCN_RISKS', 'RADII'
]
nan_rows = filtered_df[filtered_df[columns_to_replace].isna().any(axis=1)]
if not nan_rows.empty:
    print("Replacing NaN values for the following FIPS and StateName:")
    print(nan_rows[['FIPS', 'state']])

filtered_df.loc[:, columns_to_replace] = filtered_df[columns_to_replace].fillna(0)

# -------------------------------
# Step 8: Add derived columns
# -------------------------------
# 8.1: County_class
filtered_df['County_class'] = filtered_df['POPESTIMATE2022'].apply(
    lambda x: 'Metro' if x >= 50000 else 'Nonmetro'
)
# 8.2: Hurricane_risk
filtered_df['Hurricane_risk'] = filtered_df['HRCN_RISKR'].apply(
    lambda x: 'Risky' if x in ['Relatively High', 'Relatively Moderate', 'Very High'] else 'Less risky'
)
# 8.3: Wind_swath
def classify_wind_swath(radii):
    if radii == 0:
        return 'TD'
    elif radii in [34, 50]:
        return 'TS'
    elif radii == 64:
        return 'HU'
    return 'Unknown'
filtered_df['Wind_swath'] = filtered_df['RADII'].apply(classify_wind_swath)

# -------------------------------
# Step 9: Save the final DataFrame
# -------------------------------
filtered_df.to_excel(
    'C:/Users/aas0041/Desktop/eaglei_outages/01_2022_Combined_Hurricane_Data.xlsx',
    index=False
)

print("Processing completed. Updated file saved as '01_2022_Combined_Hurricane_Data.xlsx.'")










# PER CAPITA NORMALISATION OF INDEPENDENT VARIABLES 

# -------------------------------
# Step 1: Import required libraries
# -------------------------------
import pandas as pd

# -------------------------------
# Step 2: Load the input Excel file
# -------------------------------
file_path = 'C:/Users/aas0041/Desktop/eaglei_outages/01_2022_Combined_Hurricane_Data.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# -------------------------------
# Step 3: Create the new column for total critical facilities
# -------------------------------
data['Critical_facilities_Count'] = (
    data['Education_Count'] +
    data['Law_Enforcement_Count'] +
    data['Medical_Emergency_Count']
)

# -------------------------------
# Step 4: Define columns to normalize by population
# -------------------------------
columns_to_normalize = [
    'GDP', 'Critical_facilities_Count'
]

# -------------------------------
# Step 5: Normalize each column by POPESTIMATE2022
# -------------------------------
for column in columns_to_normalize:
    data[f'{column}_c'] = data[column] / data['POPESTIMATE2022']

# -------------------------------
# Step 6: Scale specific normalized columns
# -------------------------------
columns_to_scale = [
    'GDP', 'Critical_facilities_Count'
]

for column in columns_to_scale:
    if column == 'GDP':
        data[f'{column}_c'] = data[f'{column}_c'] * 1  # GDP left as is (per capita)
    else:
        data[f'{column}_c'] = data[f'{column}_c'] * 100000  # Critical facilities per 100,000 population

# -------------------------------
# Step 7: Save the updated DataFrame to a new Excel file
# -------------------------------
output_file_path = 'C:/Users/aas0041/Desktop/eaglei_outages/02_2022_Combined_Hurricane_Data_Normalized.xlsx'
data.to_excel(output_file_path, index=False)

print("Normalization and scaling completed, including 'Critical_facilities_Count', saved as '02_2022_Combined_Hurricane_Data_Normalized.xlsx'")
