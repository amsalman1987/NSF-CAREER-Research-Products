

# ===============================
# Extract rows with >100% outages
# ===============================

# Ensure numeric dtype
for col in ["max_percent_outage", "POPESTIMATE2022", "total22_customers", "sum_max_at_max_date"]:
    if col in counties.columns:
        counties[col] = pd.to_numeric(counties[col], errors="coerce")

# Subset: rows with max_percent_outage > 100
over_100 = counties[counties["max_percent_outage"] > 100].copy()

# Add count of how many times each county (by FIPS) appears
counts_by_fips = over_100.groupby("FIPS").size().rename("count")
over_100 = over_100.merge(counts_by_fips, on="FIPS", how="left")

# Compute ratios
over_100["POPESTIMATE2022_to_total22_ratio"] = np.where(
    over_100["total22_customers"] > 0,
    over_100["POPESTIMATE2022"] / over_100["total22_customers"],
    np.nan
)
over_100["sum_max_at_max_date_to_total22_ratio"] = np.where(
    over_100["total22_customers"] > 0,
    over_100["sum_max_at_max_date"] / over_100["total22_customers"],
    np.nan
)

# Select & order columns
cols_order = [
    "FIPS",
    "county",
    "count",
    "state",
    "POPESTIMATE2022",
    "total22_customers",
    "max_percent_outage",
    "sum_max_at_max_date",
    "Hurricane",
    "Year",
    "POPESTIMATE2022_to_total22_ratio",
    "sum_max_at_max_date_to_total22_ratio",
]

# Keep only columns that exist 
cols_present = [c for c in cols_order if c in over_100.columns]
over_100 = over_100[cols_present].sort_values(["max_percent_outage", "FIPS"], ascending=[False, True])

# Print summary to console and file
print_and_save("\n==================== Counties with >100% max_percent_outage ====================")
print_and_save(f"Total rows: {len(over_100)}")

# Overall unique counties (by FIPS)
if "FIPS" in over_100.columns:
    unique_counties = over_100["FIPS"].nunique()
    print_and_save(f"Unique counties (FIPS): {unique_counties}")

# Unique counties per state
state_counts_df = pd.DataFrame()
if "state" in over_100.columns and "FIPS" in over_100.columns:
    state_counts = (
        over_100.groupby("state")["FIPS"].nunique().sort_values(ascending=False)
    )
    state_counts_df = state_counts.reset_index().rename(columns={"FIPS": "unique_counties"})
    print_and_save("\nUnique counties per state (sorted):")
    for state, count in state_counts.items():
        print_and_save(f"{state}: {count}")

# Top 10 preview
if not over_100.empty:
    print_and_save("\nTop 10 (by max_percent_outage):")
    print_and_save(over_100.head(10).to_string(index=False))

# Save to Excel (two sheets)
excel_path = r"C:/Users/aas0041/Desktop/eaglei_outages/100_max_percent_outage.xlsx"
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    over_100.to_excel(writer, index=False, sheet_name=">100% Outages")
    if not state_counts_df.empty:
        state_counts_df.to_excel(writer, index=False, sheet_name="Unique per state")

print_and_save(f"\nSaved Excel file: {excel_path}")

