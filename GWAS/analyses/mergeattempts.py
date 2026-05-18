# %%
import pandas as pd
import numpy as np  
from glob import glob
from tqdm import tqdm
from scipy.stats import combine_pvalues, chi2, norm
from scipy.linalg import cholesky

import sys

def calculate_maf(a1freq):
    a2freq = 1 - a1freq
    return min(a1freq, a2freq)

def filter_HLA(df):
    hla_start = 28_477_797
    hla_end = 33_448_354
    chromosome_of_hla = 6

    return df[~((df['CHROM'] == chromosome_of_hla) & 
                (df['GENPOS'] >= hla_start) & 
                (df['GENPOS'] <= hla_end))]

def combine_pvals(pvals, method='fisher'):
    return combine_pvalues(pvals, method=method)[1]

def simes_method(pvals):
    pvals = np.sort(pvals)
    n = len(pvals)
    simes_p = np.min(pvals * n / np.arange(1, n + 1))
    return min(simes_p, 1.0)

def empirical_brown_method(pvals, corr_matrix):
    n = len(pvals)
    z_scores = norm.ppf(1 - pvals)

    L = cholesky(corr_matrix, lower=True)
    
    new_z = np.dot(L, z_scores)
    new_chi2 = np.sum(new_z**2)
    
    e_df = np.trace(corr_matrix)
    
    combined_p = 1 - norm.cdf(np.sqrt(new_chi2 / e_df))
    return combined_p

def calculate_corr_matrix(df, groupby_cols, pval_col):
    grouped = df.groupby(groupby_cols)
    pval_matrix = grouped[pval_col].apply(np.array).unstack()
    pval_matrix = pval_matrix.fillna(pval_matrix.mean())
    pval_matrix = pval_matrix.apply(norm.ppf, args=(1,))
    corr_matrix = np.corrcoef(pval_matrix, rowvar=False)
    return corr_matrix

def lancaster_method(pvals, weights=None):
    n = len(pvals)
    chi2_stats = chi2.isf(pvals, df=1)
    if weights is None:
        weights = np.ones(n)
    combined_chi2 = np.sum(weights * chi2_stats)
    combined_p = chi2.sf(combined_chi2, df=2*n)  
    return combined_p

def inverse_variance_weighted_method(betas, ses):
    weights = 1 / (ses ** 2)
    combined_beta = np.sum(weights * betas) / np.sum(weights)
    combined_se = np.sqrt(1 / np.sum(weights))
    z_score = combined_beta / combined_se
    combined_p = 2 * (1 - norm.cdf(abs(z_score)))  
    return combined_p

def harmonic_mean_pvalue(pvals):
    n = len(pvals)
    harmonic_mean = n / np.sum(1.0 / pvals)
    combined_p = min(1, harmonic_mean)
    return combined_p

# %%
# pth_sumstats = glob("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/gwas/*.gwas.regenie.gz")
pth_sumstats = glob("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/SEXINTERACT/nNs_Qntl_WBRIT_INF30_DiffAE128_5Sd_r80_discov_SexInteract_fullDSV3_output/nNs_Qntl_WBRIT_INF30_DiffAE128_5Sd_r80_discov_SexInteract_fullDSV3/results/gwas/results/gwas_ADD-INT_SNPxSex=1.0/*.gwas.regenie.gz")

# %%
df_collect = []

for pth in tqdm(pth_sumstats):
    df = pd.read_csv(pth, sep="\t")
    df['MAF'] = df['A1FREQ'].apply(calculate_maf)    
    df = df[df.MAF > 0.01]
    df = filter_HLA(df)
    df["phenotype"] = pth.split("/")[-1].split(".")[0]
    df_collect.append(df)

df = pd.concat(df_collect)
print("Concatenated!")
# df.to_pickle("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/merged_filtMAFnHLA.pkl")
df.to_pickle("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/SEXINTERACT/nNs_Qntl_WBRIT_INF30_DiffAE128_5Sd_r80_discov_SexInteract_fullDSV3_output/nNs_Qntl_WBRIT_INF30_DiffAE128_5Sd_r80_discov_SexInteract_fullDSV3/results/gwas/results/gwas_ADD-INT_SNPxSex=1.0/processed/merged_filtMAFnHLA.pkl")

# import dask.dataframe as dd
# def process_file(pth):
#     df = dd.read_csv(pth, sep="\t")
#     df['MAF'] = df['A1FREQ'].apply(calculate_maf, meta=('A1FREQ', 'float64'))
#     df = df[df.MAF > 0.01]
#     df = df.map_partitions(filter_HLA)
#     df['phenotype'] = pth.split("/")[-1].split(".")[0]
#     return df
# df_collect = [process_file(pth) for pth in tqdm(pth_sumstats)]
# df = dd.concat(df_collect)

# %%
df["P"] = 10**-df["LOG10P"]
minP = df.sort_values("P").drop_duplicates("ID", keep="first").sort_values(["CHROM", "GENPOS"])

# minP.to_csv("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/merge_attempts_minP.tsv", sep="\t", index=False)
minP.to_csv("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/SEXINTERACT/nNs_Qntl_WBRIT_INF30_DiffAE128_5Sd_r80_discov_SexInteract_fullDSV3_output/nNs_Qntl_WBRIT_INF30_DiffAE128_5Sd_r80_discov_SexInteract_fullDSV3/results/gwas/results/gwas_ADD-INT_SNPxSex=1.0/processed/merge_attempts_minP.tsv", sep="\t", index=False)
sys.exit("Done!")

# %%
try:
    fisher = df.groupby('ID')['P'].apply(lambda x: combine_pvals(x.values, method='fisher'))
    fisherP = pd.merge(minP, fisher, on=['ID'])
    fisherP.rename(columns={'P_x': 'P', 'P_y': 'fisherP'}, inplace=True)

    fisherP.to_csv("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/merge_attempts_fisherP.tsv", sep="\t", index=False)
except Exception as e:
    print(e)

# %%
try:
    stouffer = df.groupby('ID')['P'].apply(lambda x: combine_pvals(x.values, method='stouffer'))
    stoufferP = pd.merge(minP, stouffer, on=['ID'])
    stoufferP.rename(columns={'P_x': 'P', 'P_y': 'stoufferP'}, inplace=True)

    stoufferP.to_csv("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/merge_attempts_stoufferP.tsv", sep="\t", index=False)
except Exception as e:
    print(e)

# %%
try:
    lancaster = df.groupby('ID')['P'].apply(lambda x: lancaster_method(x.values))
    lancasterP = pd.merge(minP, lancaster, on=['ID'])
    lancasterP.rename(columns={'P_x': 'P', 'P_y': 'lancasterP'}, inplace=True)

    lancasterP.to_csv("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/merge_attempts_lancasterP.tsv", sep="\t", index=False)
except Exception as e:
    print(e)

# %%
try:
    simes = df.groupby('ID')['P'].apply(lambda x: simes_method(x.values))
    simesP = pd.merge(minP, simes, on=['ID'])
    simesP.rename(columns={'P_x': 'P', 'P_y': 'simesP'}, inplace=True)

    simesP.to_csv("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/merge_attempts_simesP.tsv", sep="\t", index=False)
except Exception as e:
    print(e)

# %%
try:
    harmonicMean = df.groupby('ID')['P'].apply(lambda x: harmonic_mean_pvalue(x.values))
    harmonicMeanP = pd.merge(minP, harmonicMean, on=['ID'])
    harmonicMeanP.rename(columns={'P_x': 'P', 'P_y': 'harmonicMeanP'}, inplace=True)

    harmonicMeanP.to_csv("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/merge_attempts_harmonicMeanP.tsv", sep="\t", index=False)
except Exception as e:
    print(e)

# %%
try:
    combined_pvalues = []
    for id_, group in df.groupby(['ID']):
        betas = group['BETA'].values
        ses = group['SE'].values
        combined_p = inverse_variance_weighted_method(betas, ses)
        combined_pvalues.append({'ID': id_, 'ivwP': combined_p})
    combined_pvalues_df = pd.DataFrame(combined_pvalues)
    ivwP = pd.merge(minP, combined_pvalues_df, on=['ID'])

    ivwP.to_csv("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/merge_attempts_ivwP.tsv", sep="\t", index=False) 
except Exception as e:
    print(e)

# %%
try:
    combined_pvalues = []
    for id_, group in df.groupby('ID'):
        pvals = group['P'].values
        corr_matrix = calculate_corr_matrix(df, 'ID', 'P')
        combined_p = empirical_brown_method(pvals, corr_matrix)
        combined_pvalues.append({'ID': id_, 'ebmP': combined_p})
    combined_pvalues_df = pd.DataFrame(combined_pvalues)
    ebmP = pd.merge(minP, combined_pvalues_df, on=['ID'])

    ebmP.to_csv("/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/merge_attempts_ebmP.tsv", sep="\t", index=False)
except Exception as e:
    print(e)