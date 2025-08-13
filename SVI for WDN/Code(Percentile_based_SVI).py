# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:29:48 2025

@author: mi0025
"""

import pandas as pd

# Load the data from the CSV file
file_path = ''
data = pd.read_csv(file_path)

# Set columns for themes as previously defined
theme_1_columns = ['H_mental', 'H_dia', 'H_health', 'H_hypertension', 'H_NoInsurance', 'H_kidney']
theme_2_columns = [ 'MedianAge', 
                   'PercentWithoutDiploma', 'UnemploymentPercentage', 'PercentBelowPovertyLevel', 
                   'D_fertility', 'pop18', 'D_disability', 'pop65','PerCapitaIncome']
theme_3_columns = ['Householdswithoneormorepeopleunder18', 'LackingCompletePlumbingFacilities', 
                   'RenterOccupied', 'CrowdedHouseholdPercentage', 'HousesBuiltBefore2000', 
                   'SingleParentHousehold']

# Columns that should be ranked descending because higher values indicate lower vulnerability
inverse_rank_columns = ['PerCapitaIncome']

# Apply the appropriate ranking for each variable
for column in data.columns:
    if column in inverse_rank_columns:
        data[column + '_rank'] = data[column].rank(ascending=False)  # Higher values, less vulnerability
    elif column in theme_1_columns + theme_2_columns + theme_3_columns:
        data[column + '_rank'] = data[column].rank(ascending=True)  # Higher values, more vulnerability

# Calculate percentile ranks using the formula: (Rank - 1) / (N - 1)
N = data.shape[0]  # Number of entries in the dataset
for column in data.columns:
    if '_rank' in column:
        data[column + '_percentile'] = (data[column] - 1) / (N - 1)

# Sum the percentile ranks for each theme
data['Theme_1_Percentile_Sum'] = data[[col + '_rank_percentile' for col in theme_1_columns]].sum(axis=1)
data['Theme_2_Percentile_Sum'] = data[[col + '_rank_percentile' for col in theme_2_columns]].sum(axis=1)
data['Theme_3_Percentile_Sum'] = data[[col + '_rank_percentile' for col in theme_3_columns]].sum(axis=1)

# Calculate the overall percentile rank by summing up the percentile rankings of the three domains
data['Overall_Percentile_Rank'] = data[['Theme_1_Percentile_Sum', 'Theme_2_Percentile_Sum', 'Theme_3_Percentile_Sum']].sum(axis=1)

# Normalize the Overall_Percentile_Rank to a 0-1 scale
data['Normalized_Overall_Percentile_Rank'] = (data['Overall_Percentile_Rank'] - data['Overall_Percentile_Rank'].min()) / (data['Overall_Percentile_Rank'].max() - data['Overall_Percentile_Rank'].min())

# Save the result to a new CSV file
output_path = ''
data[['FIPS', 'Theme_1_Percentile_Sum', 'Theme_2_Percentile_Sum', 'Theme_3_Percentile_Sum', 'Normalized_Overall_Percentile_Rank']].to_csv(output_path, index=False)

# Confirm the path to the saved file and show the first few rows as a preview
output_path, data[['FIPS', 'Theme_1_Percentile_Sum', 'Theme_2_Percentile_Sum', 'Theme_3_Percentile_Sum', 'Normalized_Overall_Percentile_Rank']].head() 