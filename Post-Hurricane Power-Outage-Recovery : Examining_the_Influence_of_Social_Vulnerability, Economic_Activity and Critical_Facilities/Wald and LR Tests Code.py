

from spreg import ML_Lag 
from scipy.stats import chi2
import numpy as np

# --- Assume y, X_combined, X_final_sdm, w_combined, column_names_sdm, column_names_original are already defined ---

# ---------------------------------------------------------
# Step 1: Fit SDM and SAR Models
# ---------------------------------------------------------
model_sdm = ML_Lag(y, X_final_sdm, w_combined, name_y="y", name_x=column_names_sdm, name_w="Combined Weights")
model_sar = ML_Lag(y, X_combined, w_combined, name_y="y", name_x=column_names_original, name_w="Combined Weights")

# ---------------------------------------------------------
# Step 2: Wald Test for Spatially Lagged Coefficients (SDM)
# ---------------------------------------------------------
n_orig = len(column_names_original)
n_total = len(column_names_sdm)
n_spatial = n_total - n_orig
beta_sdm = model_sdm.betas
var_beta_sdm = model_sdm.vm
beta_spatial = beta_sdm[n_orig:]
var_beta_spatial = var_beta_sdm[n_orig:, n_orig:]
wald_stat = float(beta_spatial.T @ np.linalg.inv(var_beta_spatial) @ beta_spatial)
wald_p = chi2.sf(wald_stat, df=n_spatial)

# ---------------------------------------------------------
# Step 3: Likelihood Ratio (LR) Test: (SDM vs. SAR)
# ---------------------------------------------------------
LR_stat = max(0, 2 * (model_sdm.logll - model_sar.logll))
df_diff = abs(model_sdm.k - model_sar.k)
p_value = chi2.sf(LR_stat, df=df_diff)

# ---------------------------------------------------------
# Step 4: Print and Save Results
# ---------------------------------------------------------
output_file = "Wald and LR Tests Results.txt"
with open(output_file, "w") as f:
    
    # WALD TEST
    print("\n" + "-"*50)
    print(" WALD TEST: Significance of All Spatially Lagged Coefficients (SDM) ")
    print("-"*50)
    print(f"Wald test statistic: {wald_stat:.4f}, p-value: {wald_p:.4f}")
    # Save to file
    f.write("\n" + "-"*50 + "\n")
    f.write(" WALD TEST: Significance of All Spatially Lagged Coefficients (SDM) \n")
    f.write("-"*50 + "\n")
    f.write(f"Wald test statistic: {wald_stat:.4f}, p-value: {wald_p:.4f}\n")
    
    # LR TEST
    print("\n" + "-"*50)
    print(" LIKELIHOOD RATIO (LR) TEST: (SDM vs SAR) ")
    print("-"*50)
    print(f"LR test statistic: {LR_stat:.4f}, df: {df_diff}, p-value: {p_value:.4f}")
    # Save to file
    f.write("\n" + "-"*50 + "\n")
    f.write(" LIKELIHOOD RATIO (LR) TEST: (SDM vs SAR) \n")
    f.write("-"*50 + "\n")
    f.write(f"LR test statistic: {LR_stat:.4f}, df: {df_diff}, p-value: {p_value:.4f}\n")

print(f"\nResults have been saved to '{output_file}'")
