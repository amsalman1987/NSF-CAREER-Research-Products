




import geopandas as gpd

# --------------------------------------------------------
# Step 1: Load the county and wind swath shapefiles
# --------------------------------------------------------
counties = gpd.read_file('C:/Users/aas0041/Downloads/tl_2023_us_county/tl_2023_us_county.shp')
wind_swath = gpd.read_file('C:/Users/aas0041/Downloads/al092020_best_track/AL092020_windswath.shp')

# --------------------------------------------------------
# Step 2: Select only relevant columns from counties
# --------------------------------------------------------
counties = counties[["STATEFP", "GEOID", "NAME", "NAMELSAD", "NGEOID", "geometry"]]

# --------------------------------------------------------
# Step 3: Reproject both layers to EPSG:5070 (Albers Equal Area)
# --------------------------------------------------------
counties = counties.to_crs(epsg=5070)
wind_swath = wind_swath.to_crs(epsg=5070)

# --------------------------------------------------------
# Step 4: Intersect counties and wind swath to find overlapping areas
# --------------------------------------------------------
overlapping_counties = gpd.overlay(counties, wind_swath, how="intersection")

# --------------------------------------------------------
# Step 5: Save results as a shapefile and as a CSV (attributes only)
# --------------------------------------------------------
overlapping_counties.to_file("Beryl_Wind_Speed_epsg5070.shp")
overlapping_counties.drop(columns="geometry").to_csv("overlapping_cccounties_epsg5070.csv", index=False)

# --------------------------------------------------------
# Step 6: Print the first few rows to verify output
# --------------------------------------------------------
print(overlapping_counties.head())











