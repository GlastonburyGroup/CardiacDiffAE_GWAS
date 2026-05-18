# %%
import os
import pandas as pd
from glob import glob
from tqdm import tqdm

# %%
pth_cond = "/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/gwas/independent/genome-wide_significant_hits_post_cojo.csv"
root_pth_sumstats = "/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/gwas"
out_pth = "/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30"

# %%
pth_sumstats = glob(f"{root_pth_sumstats}/*.gwas.regenie.gz")
os.makedirs(out_pth, exist_ok=True)

# %%
cond_ind = pd.read_csv(pth_cond)

df0 = pd.read_table(pth_sumstats[0], compression="gzip")
df0 = df0.rename(columns={"CHROM":"Chr", "GENPOS":"bp"})
df0 = df0[["Chr", "bp", "ID"]]
cond_df0 = df0.merge(cond_ind, on=["Chr", "bp"], how="inner")
cond_ind_SNPs = cond_df0.ID.tolist()
print(f"{len(cond_ind_SNPs)} SNPs found after the conditional analysis")

with open(f"{out_pth}/cond_SNPs_postCOJO.txt", "w") as f:
    for snp in cond_ind_SNPs:
        f.write(f"{snp}\n")

# %%
sig_SNPs = []
for pth in tqdm(pth_sumstats):
    df = pd.read_table(pth, compression="gzip")
    df = df[df['A1FREQ'] > 0.01]  
    df_sig = df[df["LOG10P"] > 7.3]
    sig_SNPs += df_sig.ID.tolist()
sig_SNPs = list(set(sig_SNPs))
n_sig_SNPs = len(sig_SNPs)
print(f"{n_sig_SNPs} uique SNPs found with MLOG10P > 7.3 and A1FREQ > 0.01")

# %%
final_SNPs = list(set(sig_SNPs+cond_ind_SNPs))
print(f"{len(final_SNPs)} unique SNPs found in the final list, {len(final_SNPs)-n_sig_SNPs} SNPs were added from the conditional analysis")

# %%
with open(f"{out_pth}/sig_plus_cond_SNPs.txt", "w") as f:
    for snp in final_SNPs:
        f.write(f"{snp}\n")