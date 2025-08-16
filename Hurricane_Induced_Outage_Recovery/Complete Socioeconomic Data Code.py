


                        # COMPILING SOCIOECONOMIC DATA FOR SVI CONSTRUCTION


import pandas as pd

# ----------------------------------------------------
# Step 1: Load the base county file (master file)
# ----------------------------------------------------
base_file = 'C:/Users/aas0041/Desktop/OutageData/AbrFileGeo.csv'
df = pd.read_csv(base_file, encoding='ISO-8859-1')

# ----------------------------------------------------
# Step 2: Merge animal/health/census data data files
# ----------------------------------------------------
merge_plan = [
    # (filename, columns_to_add, left_on, right_on)
    ('Goats 2022.csv',              ['State ANSIG', 'County ANSIG', 'Goats Value'],      ['STATEFP', 'COUNTYFP'], ['State ANSIG', 'County ANSIG']),
    ('Sheep Inc Lambs 2022.csv',    ['State ANSIS', 'County ANSIS', 'Sheep Value'],      ['STATEFP', 'COUNTYFP'], ['State ANSIS', 'County ANSIS']),
    ('Hogs 2022.csv',               ['State ANSIH', 'County ANSIH', 'Hogs Value'],       ['STATEFP', 'COUNTYFP'], ['State ANSIH', 'County ANSIH']),
    ('Poultry Total 2022.csv',      ['State ANSIP', 'County ANSIP', 'Poultry Value'],    ['STATEFP', 'COUNTYFP'], ['State ANSIP', 'County ANSIP']),
    ('Cattle Incl Calves 2022.csv', ['State ANSIC', 'County ANSIC', 'Cattle Value'],     ['STATEFP', 'COUNTYFP'], ['State ANSIC', 'County ANSIC']),
    ('Electricity medicare dependents.csv', ['FIPS_Code', 'At risk percentage'],          'GEOID', 'FIPS_Code'),
    ('PLACES__County_Data__GIS.csv',        ['CountyFIPS', 'CASTHMA_CrudePrev', 'CHD_CrudePrev', 'DIABETES_CrudePrev'], 'GEOID', 'CountyFIPS'),
    ('Underlying Cause of Death, 2018-2022, Diabetes.csv', ['County Code', 'Diabetes death'], 'GEOID', 'County Code'),
    ('Underlying Cause of Death, 2018-2022, Disease of heart.csv', ['County Code', 'Heart D death'], 'GEOID', 'County Code'),
    ('Underlying Cause of Death, 2018-2022, Asthma.csv', ['County Code', 'Asthma death'], 'GEOID', 'County Code'),
    ('2020 Nursing home res.csv',   ['Geographic Area', 'Nurs Hom'],                      'Geographic Area', 'Geographic Area'),
    ('2022 age 5.csv',              ['Geographic Area', 'Under 5 yrs'],                   'Geographic Area', 'Geographic Area'),
    ('2022 age 65.csv',             ['Geographic Area', 'Over 65 yrs'],                   'Geographic Area', 'Geographic Area'),
    ('2022 ambulry diff.csv',       ['Geographic Area', 'Amblry diffclty'],               'Geographic Area', 'Geographic Area'),
    ('2022 child under 6.csv',      ['Geographic Area', 'Child under 6 yrs'],             'Geographic Area', 'Geographic Area'),
    ('2022 child under 18.csv',     ['Geographic Area', 'Child under 18yrs'],             'Geographic Area', 'Geographic Area'),
    ('2022 disability.csv',         ['Geographic Area', 'Disability'],                    'Geographic Area', 'Geographic Area'),
    ('2022 educ job.csv',           ['Geographic Area', 'Educ job'],                      'Geographic Area', 'Geographic Area'),
    ('2022 educ no diplm.csv',      ['Geographic Area', 'No diploma'],                    'Geographic Area', 'Geographic Area'),
    ('2022 health job.csv',         ['Geographic Area', 'Health job'],                    'Geographic Area', 'Geographic Area'),
    ('2022 HH size.csv',            ['Geographic Area', 'HH size'],                       'Geographic Area', 'Geographic Area'),
    ('2022 Lvn alone.csv',          ['Geographic Area', 'Lvng alone'],                    'Geographic Area', 'Geographic Area'),
    ('2022 med age.csv',            ['Geographic Area', 'Median age '],                   'Geographic Area', 'Geographic Area'),
    ('2022 med incm.csv',           ['Geographic Area', 'Median income'],                 'Geographic Area', 'Geographic Area'),
    ('2022 natural rescs occp.csv', ['Geographic Area', 'Natural rescs'],                 'Geographic Area', 'Geographic Area'),
    ('2022 no vehicle.csv',         ['Geographic Area', 'No vhcl'],                       'Geographic Area', 'Geographic Area'),
    ('2022 povrty lvl.csv',         ['Geographic Area', 'Povrty lvl'],                    'Geographic Area', 'Geographic Area'),
    ('2022 production occp.csv',    ['Geographic Area', 'Production job'],                'Geographic Area', 'Geographic Area'),
    ('2022 service occp.csv',       ['Geographic Area', 'Service job'],                   'Geographic Area', 'Geographic Area'),
    ('2022 unit struc 2 or more.csv', ['Geographic Area', '2unit or more'],               'Geographic Area', 'Geographic Area'),
]
svi_folder = 'C:/Users/aas0041/Desktop/Datasets/SVI AllCounty Data/'

for filename, cols, left_on, right_on in merge_plan:
    print(f"Merging: {filename}")
    try:
        to_merge = pd.read_csv(svi_folder + filename, encoding='ISO-8859-1')
        df = pd.merge(df, to_merge[cols], left_on=left_on, right_on=right_on, how='left')
    except Exception as e:
        print(f"Error merging {filename}: {e}")


# ----------------------------------------------------
# Step 3: Merge labor, GDP, and critical facilities data
# ----------------------------------------------------


# ----------------------------------------------------
# Step 3.1: Merge Labor Data (Establishment, Employment, Total Wages)
# ----------------------------------------------------
labour_file = 'C:/Users/aas0041/Desktop/OutageData/XLabour data 2023.csv'
labour_df = pd.read_csv(labour_file, encoding='ISO-8859-1')
df = pd.merge(
    df,
    labour_df[['AreaCode', 'Establishment', 'Employment', 'Total Wages']],
    left_on='GEOID',
    right_on='AreaCode',
    how='left'
)

# ----------------------------------------------------
# Step 3.2: Merge GDP Data
# ----------------------------------------------------
gdp_file = 'C:/Users/aas0041/Desktop/OutageData/XGDP All 2022.csv'
gdp_df = pd.read_csv(gdp_file, encoding='ISO-8859-1')
gdp_df['GeoFIPS'] = gdp_df['GeoFIPS'].str.replace('"', '').str.strip().astype('int64')
df = pd.merge(
    df,
    gdp_df[['GeoFIPS', 'GDP']],
    left_on='GEOID',
    right_on='GeoFIPS',
    how='left'
)

# ----------------------------------------------------
# Step 3.3: Merge Critical Facilities Data (Education, Law Enforcement, Medical & Emergency) structures
# ----------------------------------------------------
critical_file = 'C:/Users/aas0041/Desktop/OutageData/XNational_structures_SpatialJoin_TableToExcel2.csv'
crit_df = pd.read_csv(critical_file, encoding='ISO-8859-1', low_memory=False)

# Define structure categories
medical_codes = [80012, 74001, 74026]
law_enforcement_codes = [74034, 74036]
education_ftype = 730

# Assign category labels
crit_df['Structure_Type'] = None
crit_df.loc[crit_df['FCODE'].isin(medical_codes), 'Structure_Type'] = 'Medical & Emergency'
crit_df.loc[crit_df['FCODE'].isin(law_enforcement_codes), 'Structure_Type'] = 'Law Enforcement'
crit_df.loc[crit_df['FTYPE'] == education_ftype, 'Structure_Type'] = 'Education'

# Count and pivot
summary = crit_df[crit_df['Structure_Type'].notnull()] \
    .groupby(['FIPS', 'Structure_Type']).size().reset_index(name='Count')
pivot = summary.pivot(index='FIPS', columns='Structure_Type', values='Count').reset_index()
pivot.columns.name = None
pivot = pivot.rename(columns={
    'Education': 'Education_Count',
    'Law Enforcement': 'Law_Enforcement_Count',
    'Medical & Emergency': 'Medical_Emergency_Count'
})

# Merge with base DataFrame (use GEOID = FIPS)
df = pd.merge(
    df,
    pivot,
    left_on='GEOID',
    right_on='FIPS',
    how='left'
)

# ----------------------------------------------------
# Step 3.4: Merge Hurricane Risk Data (NRI risk scores)
# ----------------------------------------------------
risk_file = 'C:/Users/aas0041/Desktop/OutageData/NRI_Table_Counties.csv'
risk_df = pd.read_csv(risk_file, encoding='ISO-8859-1')
df = pd.merge(
    df,
    risk_df[['STCOFIPS', 'HRCN_RISKS', 'HRCN_RISKR']],
    left_on='GEOID',
    right_on='STCOFIPS',
    how='left'
)


# ----------------------------------------------------
# Step 3.5: Merge Urban-Rural Codes (NCHSUR 2013 classification)
# ----------------------------------------------------
nchs_file = 'C:/Users/aas0041/Desktop/Datasets/SVI AllCounty Data/NCHSURCodes2013.csv'
nchs_df = pd.read_csv(nchs_file, encoding='ISO-8859-1')

df = pd.merge(
    df,
    nchs_df[['FIPS code', '2013 code']],
    left_on='GEOID',
    right_on='FIPS code',
    how='left'
)


# ----------------------------------------------------
# Step 4: Create animal "per 1000 population" columns
# ----------------------------------------------------
for col in ['Goats Value', 'Sheep Value', 'Hogs Value', 'Poultry Value', 'Cattle Value', 'POPESTIMATE2022']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for value_col, per_col in [
    ('Goats Value',   'Goats per 1000'),
    ('Sheep Value',   'Sheep per 1000'),
    ('Hogs Value',    'Hogs per 1000'),
    ('Poultry Value', 'Poultry per 1000'),
    ('Cattle Value',  'Cattle per 1000')
]:
    if value_col in df.columns:
        df[per_col] = df[value_col] / df['POPESTIMATE2022'] * 1000


# ----------------------------------------------------
# Step 5: Filter by states before saving
# ----------------------------------------------------
states_to_extract = [
    'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont',
    'New Jersey', 'New York', 'Pennsylvania', 'Delaware', 'Florida', 'Georgia', 'Maryland',
    'North Carolina', 'South Carolina', 'Virginia', 'West Virginia',
    'Alabama', 'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas', 'Louisiana', 'Texas',
    'Illinois', 'Indiana', 'Ohio', 'Missouri', 'Oklahoma'
]
state_abbr_to_extract = [
    'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA', 'DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV',
    'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'TX',
    'IL', 'IN', 'OH', 'MO', 'OK'
]
filtered_df = df[
    (df['StateName'].isin(states_to_extract)) &
    (df['STATE ABR'].isin(state_abbr_to_extract))
]

# ----------------------------------------------------
# Step 6: Save only the filtered states to a new CSV
# ----------------------------------------------------
filtered_df.to_csv('C:/Users/aas0041/Desktop/Journal Data/Compiled_Socioeconomic_Data.csv', index=False)
print("Filtered data saved to 'Compiled_Socioeconomic_Data.csv'")



# Clean the final file of all Nan values before moving to SVI construction
# And loving county, Texas is removed from  due to some missing Census data







                        # ADDING THE CALCULATED SVI VALUES TO THE COMPILED SOCIOECONOMIC DATA FILE 


import pandas as pd

# ----------------------------------------------------
# Step 1: Define file paths and theme columns
# ----------------------------------------------------
file_paths = [
    ('C:/Users/aas0041/Desktop/OutageData/Allnormalized_pareto_H_P_E themes.csv', 'All_SVI'),
    ('C:/Users/aas0041/Desktop/OutageData/Allnormalized_pareto_H_P themes.csv', 'HealthPrep_SVI'),
    ('C:/Users/aas0041/Desktop/OutageData/Allnormalized_pareto_H_E themes.csv', 'HealthEvac_SVI'),
    ('C:/Users/aas0041/Desktop/OutageData/Allnormalized_pareto_P_E themes.csv', 'PrepEvac_SVI'),
    ('C:/Users/aas0041/Desktop/OutageData/Allnormalized_pareto_H themes.csv', 'Health_SVI'),
    ('C:/Users/aas0041/Desktop/OutageData/Allnormalized_pareto_P themes.csv', 'Prep_SVI'),
    ('C:/Users/aas0041/Desktop/OutageData/Allnormalized_pareto_E themes.csv', 'Evac_SVI'),
]
main_file = 'C:/Users/aas0041/Desktop/Journal Data/Compiled_Socioeconomic_Data.csv'

# ----------------------------------------------------
# Step 2: Load the base DataFrame 
# ----------------------------------------------------
merged_df = pd.read_csv(main_file, encoding='ISO-8859-1')

# ----------------------------------------------------
# Step 3: Loop through each theme file and merge by 'Geographic Area'
# ----------------------------------------------------
for file_path, column_name in file_paths:
    theme_df = pd.read_csv(file_path, encoding='ISO-8859-1')[['Geographic Area', column_name]]
    merged_df = pd.merge(
        merged_df,
        theme_df,
        on='Geographic Area',
        how='left'
    )

# ----------------------------------------------------
# Step 4: Save the merged DataFrame
# ----------------------------------------------------
output_path = 'C:/Users/aas0041/Desktop/Journal Data/Complete_Socioeconomic_Data.csv'
merged_df.to_csv(output_path, index=False)
print("Merge completed and saved as 'Complete_Socioeconomic_Data.csv'")





















