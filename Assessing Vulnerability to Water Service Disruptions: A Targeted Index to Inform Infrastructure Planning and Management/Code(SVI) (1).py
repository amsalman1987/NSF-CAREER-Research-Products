# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 23:57:39 2025

@author: mi0025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from factor_analyzer import Rotator


# Read the Excel file
file_path = ''
df = pd.read_csv(file_path)

# Classify columns according to themes
# Set columns for themes as previously defined
theme_1_columns = ['H_mental', 'H_dia', 'H_health', 'H_hypertension', 'H_NoInsurance', 'H_kidney'] # heath related
theme_2_columns = [ 'MedianAge', 
                   'PercentWithoutDiploma', 'UnemploymentPercentage', 'PercentBelowPovertyLevel', 
                   'D_fertility', 'pop18', 'D_disability', 'pop65','PerCapitaIncome',] # socioeconomic demograpghic
theme_3_columns = ['Householdswithoneormorepeopleunder18', 'LackingCompletePlumbingFacilities', 
                   'RenterOccupied', 'CrowdedHouseholdPercentage', 'HousesBuiltBefore2000', 
                   'SingleParentHousehold']                                                       # Infrastructure 



# Columns to calculate complements for themes
theme_1_complement_columns = []
theme_2_complement_columns = ['PerCapitaIncome']
theme_3_complement_columns = []

# Ensure the 'COUNTY' column is included
county_column = ['FIPS']


# Function to normalize data
def normalize(df, columns):
    # Ensure that the specified columns are cast to float type
    df[columns] = df[columns].astype(float)
    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    df.loc[:, columns] = scaler.fit_transform(df.loc[:, columns])
    return df


# Function to calculate complements and update normalized data
def apply_complements(df, complement_columns):
    if complement_columns:
        #print("Data Before Complement (first 5 rows):")
        #print(df[complement_columns].head())
        df.loc[:, complement_columns] = 1 - df.loc[:, complement_columns]
        #print("\nData After Complement (first 5 rows):")
        #print(df[complement_columns].head())
    return df


# Create and normalize separate DataFrames for each theme

theme_1_df = df[county_column + theme_1_columns].copy()
theme_2_df = df[county_column + theme_2_columns].copy()
theme_3_df = df[county_column + theme_3_columns].copy()

# Normalize and apply complements for each theme

theme_1_df = normalize(theme_1_df, theme_1_columns)
theme_1_df = apply_complements(theme_1_df, theme_1_complement_columns)


theme_2_df = normalize(theme_2_df, theme_2_columns)
theme_2_df = apply_complements(theme_2_df, theme_2_complement_columns)


theme_3_df = normalize(theme_3_df, theme_3_columns)
theme_3_df = apply_complements(theme_3_df, theme_3_complement_columns)



# Function to perform PCA
def perform_pca(df, columns, theme_name):
    pca = PCA()  # No n_components specified, PCA will generate all components
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
    selected_eigenvalues = eigenvalues[:num_components]
    selected_pca_df = pca_df.iloc[:, :num_components]
    
    # Calculate total variance explained by selected components
    total_variance_selected = np.sum(selected_variance_ratio)
    total_variance_percentage = total_variance_selected * 100

    # Print results for PCA and selected components with theme name
    print(f"\n{theme_name} - Initial Principal Components (All):")
    print(pca_df)
    print(f"\n{theme_name} - Initial Explained Variance Ratio (All):", explained_variance_ratio)
    print(f"{theme_name} - Initial Eigenvalues (All):", eigenvalues)
    print(f"{theme_name} - Number of Selected Components: {num_components}")
    print(f"\n{theme_name} - Selected Principal Components:")
    print(selected_pca_df)
    print(f"{theme_name} - Selected Explained Variance Ratio:", selected_variance_ratio)
    print(f"{theme_name} - Selected Eigenvalues:", selected_eigenvalues)
    print(f"{theme_name} - Sum of Selected Variance Ratios: {total_variance_selected:.4f}")
    print(f"{theme_name} - Percentage of Variance Explained by Selected Components: {total_variance_percentage:.2f}%")

    return selected_pca_df, explained_variance_ratio, eigenvalues, loadings_df


def apply_varimax_with_kaiser_normalization(loadings_df):
    kaiser_normalized = loadings_df.div(np.sqrt((loadings_df**2).sum(axis=0)), axis=1)
    rotator = Rotator(method='varimax')
    rotated_loadings = rotator.fit_transform(kaiser_normalized)
    scaling_factors = np.sqrt((loadings_df**2).sum(axis=0)).values
    rotated_loadings = rotated_loadings * scaling_factors[:, np.newaxis]
    rotated_loadings_df = pd.DataFrame(rotated_loadings, index=loadings_df.index, columns=loadings_df.columns)
    return rotated_loadings_df


# Function to calculate L2 norm for each principal component
def calculate_l2_norm(pca_df):
    l2_norms = np.linalg.norm(pca_df.values, axis=1)
    return l2_norms


# Function to perform Pareto ranking with the dominance criteria
def pareto_ranking(norms_df, columns):
    # Initialize ranking dictionary
    ranks = {}
    rank = 1
    remaining_counties = norms_df[['FIPS'] + columns].copy()
    
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
            ranks[row['FIPS']] = (rank, *row[columns].values)
            remaining_counties = remaining_counties[remaining_counties['FIPS'] != row['FIPS']]
        
        rank += 1
    
    return ranks

# Perform PCA for each theme
theme_1_pca_df, theme_1_explained_variance, theme_1_eigenvalues, theme_1_loadings = perform_pca(theme_1_df, theme_1_columns, "Theme 1")
theme_2_pca_df, theme_2_explained_variance, theme_2_eigenvalues, theme_2_loadings = perform_pca(theme_2_df, theme_2_columns, "Theme 2")
theme_3_pca_df, theme_3_explained_variance, theme_3_eigenvalues, theme_3_loadings = perform_pca(theme_3_df, theme_3_columns, "Theme 3")

# Apply Varimax rotation with Kaiser normalization to each theme
theme_1_rotated_loadings = apply_varimax_with_kaiser_normalization(theme_1_loadings)
theme_2_rotated_loadings = apply_varimax_with_kaiser_normalization(theme_2_loadings)
theme_3_rotated_loadings = apply_varimax_with_kaiser_normalization(theme_3_loadings)


# Calculate L2 norms for each theme's principal components
theme_1_l2_norms = calculate_l2_norm(theme_1_pca_df)
theme_2_l2_norms = calculate_l2_norm(theme_2_pca_df)
theme_3_l2_norms = calculate_l2_norm(theme_3_pca_df)


# Combine L2 norms into a single DataFrame
l2_norms_df = pd.DataFrame({
    'FIPS': df['FIPS'],
    'Theme_1_L2': theme_1_l2_norms,
    'Theme_2_L2': theme_2_l2_norms,
    'Theme_3_L2': theme_3_l2_norms
})


# Get the Pareto rankings with modified dominance criteria
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
                                              columns=['FIPS', 'Rank', 'Theme_1_L2', 'Theme_2_L2', 'Theme_3_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 1 and 2'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_1_2.items()],
                                                 columns=['FIPS', 'Rank', 'Theme_1_L2', 'Theme_2_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 1 and 3'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_1_3.items()],
                                                 columns=['FIPS', 'Rank', 'Theme_1_L2', 'Theme_3_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 2 and 3'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_2_3.items()],
                                                 columns=['FIPS', 'Rank', 'Theme_2_L2', 'Theme_3_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 1'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_1.items()],
                                           columns=['FIPS', 'Rank', 'Theme_1_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 2'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_2.items()],
                                           columns=['FIPS', 'Rank', 'Theme_2_L2']).sort_values(by='Rank')
pareto_ranks_dfs['Theme 3'] = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_3.items()],
                                           columns=['FIPS', 'Rank', 'Theme_3_L2']).sort_values(by='Rank')


# Initialize the MinMaxScaler for normalization
scaler = MinMaxScaler()

# Loop through each Pareto rank DataFrame to calculate reversed and normalized ranks
for theme_key, df in pareto_ranks_dfs.items():
    # Calculate the reversed rank by subtracting from the maximum rank + 1
    df['Reversed Rank'] = df['Rank'].max() + 1 - df['Rank']
    
    # Normalize the reversed rank between 0 and 1
    df['Normalized Reversed Rank'] = scaler.fit_transform(df[['Reversed Rank']])
    
    # Update the DataFrame in the dictionary
    pareto_ranks_dfs[theme_key] = df

# Now pareto_ranks_dfs contains DataFrames for each theme with reversed and normalized ranks


# Print the results
for key, df in pareto_ranks_dfs.items():
    print(f"{key} Pareto Ranking:")
    print(df)  # Display the top 10 ranked counties for each ranking
    print()


# Save all Pareto rank DataFrames to an Excel file with separate sheets
output_file_path = 'Alllnormalized_pareto_ranks_all_themes.xlsx'

# Use ExcelWriter to save each DataFrame as a sheet in the same file
with pd.ExcelWriter(output_file_path) as writer:
    for theme_key, df in pareto_ranks_dfs.items():
        # Select relevant columns to save
        df[['FIPS', 'Rank', 'Reversed Rank', 'Normalized Reversed Rank']].to_excel(writer, sheet_name=theme_key, index=False)


# Print results for Theme 1 with Varimax rotation and L2 norms
print("\nTheme 1 Selected Components - Varimax Rotated Loadings:\n", theme_1_rotated_loadings)
print("\nTheme 1 Selected PCA DataFrame (First 5 Rows):\n", theme_1_pca_df.head())
print("\nTheme 1 Selected Explained Variance Ratio:", theme_1_explained_variance[:len(theme_1_pca_df.columns)])
print("\nTheme 1 Selected Eigenvalues:", theme_1_eigenvalues[:len(theme_1_pca_df.columns)])
print("\nTheme 1 L2 Norms of Selected Principal Components:\n", theme_1_l2_norms)

# Similarly for Theme 2
print("\nTheme 2 Selected Components - Varimax Rotated Loadings:\n", theme_2_rotated_loadings)
print("\nTheme 2 Selected PCA DataFrame (First 5 Rows):\n", theme_2_pca_df.head())
print("\nTheme 2 Selected Explained Variance Ratio:", theme_2_explained_variance[:len(theme_2_pca_df.columns)])
print("\nTheme 2 Selected Eigenvalues:", theme_2_eigenvalues[:len(theme_2_pca_df.columns)])
print("\nTheme 2 L2 Norms of Selected Principal Components:\n", theme_2_l2_norms)

# Similarly for Theme 3
print("\nTheme 3 Selected Components - Varimax Rotated Loadings:\n", theme_3_rotated_loadings)
print("\nTheme 3 Selected PCA DataFrame (First 5 Rows):\n", theme_3_pca_df.head())
print("\nTheme 3 Selected Explained Variance Ratio:", theme_3_explained_variance[:len(theme_3_pca_df.columns)])
print("\nTheme 3 Selected Eigenvalues:", theme_3_eigenvalues[:len(theme_3_pca_df.columns)])
print("\nTheme 3 L2 Norms of Selected Principal Components:\n", theme_3_l2_norms)

# Print combined L2 norms
print("\nCombined L2 Norms of Selected Principal Components Across All Themes:\n", l2_norms_df)
