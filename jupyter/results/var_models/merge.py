import pandas as pd
from pathlib import Path

p1 = pd.read_csv("4_mod_psnrout_p1.csv")
p2 = pd.read_csv("4_mod_psnrout_p2.csv")

cutoff = 6000

p1_cut = p1[p1["Step"] <= cutoff]
p2_cut = p2[p2["Step"] > cutoff]

merged = pd.concat([p1_cut, p2_cut], ignore_index=True)
merged = merged.sort_values("Step").reset_index(drop=True)

merged.to_csv("4_mod_psnrin_merged.csv", index=False)
