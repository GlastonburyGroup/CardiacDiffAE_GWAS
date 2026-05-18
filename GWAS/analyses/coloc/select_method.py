import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

merge_corrs = True
do_norm = True
isDiffAE = False
only_corr = True

df = pd.read_csv("/group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/analyses/coloc/SelectMethod.csv", delimiter=';')
df = df.dropna(axis=1, how='all')
df = df.dropna()

if isDiffAE:
    df = df[df['Model'].str.contains('DiffAE')]
else:
    df = df[~df['Model'].str.contains('DiffAE')]
# Normalising the columns to bring them to a comparable scale
scaler = MinMaxScaler()

if merge_corrs:
    df["n_corr_str"] = df["n_corr_str_body"] + df["n_corr_str_heart"]
    df["n_corr_mod"] = df["n_corr_mod_body"] + df["n_corr_mod_heart"]
    if do_norm:
        df[['n_ind_sigs', 'n_colocs', 'n_corr_str', 'n_corr_mod']] = scaler.fit_transform(
            df[['n_ind_sigs', 'n_colocs', 'n_corr_str', 'n_corr_mod']])
elif do_norm:
    df[['n_ind_sigs', 'n_colocs', 'n_corr_str_body', 'n_corr_mod_body', 'n_corr_str_heart', 'n_corr_mod_heart']] = scaler.fit_transform(
        df[['n_ind_sigs', 'n_colocs', 'n_corr_str_body', 'n_corr_mod_body', 'n_corr_str_heart', 'n_corr_mod_heart']])

weights = {
    'independent_signals': 0.4,
    'colocalised_signals': 0.3,
    'strong_correlations': 0.2,
    'moderate_correlations': 0.1
}

if only_corr:
    if merge_corrs:
        df['weighted_score'] = (
                                df['n_corr_str'] * weights['strong_correlations'] +
                                df['n_corr_mod'] * weights['moderate_correlations'])  
    else:
        df['weighted_score'] = (
                                df['n_corr_str_body'] * weights['strong_correlations'] +
                                df['n_corr_mod_body'] * weights['moderate_correlations'] +
                                df['n_corr_str_heart'] * weights['strong_correlations'] +
                                df['n_corr_mod_heart'] * weights['moderate_correlations'])
else:
    if merge_corrs:
        df['weighted_score'] = (df['n_ind_sigs'] * weights['independent_signals'] + 
                                df['n_colocs'] * weights['colocalised_signals'] +
                                df['n_corr_str'] * weights['strong_correlations'] +
                                df['n_corr_mod'] * weights['moderate_correlations'])  
    else:
        df['weighted_score'] = (df['n_ind_sigs'] * weights['independent_signals'] + 
                                df['n_colocs'] * weights['colocalised_signals'] +
                                df['n_corr_str_body'] * weights['strong_correlations'] +
                                df['n_corr_mod_body'] * weights['moderate_correlations'] +
                                df['n_corr_str_heart'] * weights['strong_correlations'] +
                                df['n_corr_mod_heart'] * weights['moderate_correlations'])
        

# Ranking the methods based on the weighted score
df.sort_values(by='weighted_score', ascending=False, inplace=True)

# Display the ranked methods
print("Ranked Methods:----------------\n")
print(df[['Metric', 'Model', 'weighted_score']])

model_median = df.groupby('Model')['weighted_score'].median()
model_median.sort_values(ascending=False, inplace=True)
print("\nRanked Models:----------------\n")
print(model_median)

metric_median = df.groupby('Metric')['weighted_score'].median()
metric_median.sort_values(ascending=False, inplace=True)
print("\nRanked Metric:----------------\n")
print(metric_median)