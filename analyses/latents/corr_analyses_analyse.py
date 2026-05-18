import pandas as pd

def get_counts(df, columns, filter_column, filter_value):
    df = df[df.Strength.isin(["Strong", "Moderate"])]
    return df[df[filter_column].str.contains(filter_value, na=False)].groupby(columns)['UniqueAttributeCount'].sum().unstack(fill_value=0)

def split_method_model(model_name, methods_merge):
    for method in methods_merge:
        if method in model_name:
            return method, model_name.replace(method + "_", "", 1)
    return None, model_name

def get_value(df, index, model, strength):
    try:
        value = df.loc[index, model][strength]
    except KeyError:
        value = ''  # or 'X' or some other indicator of missing data
    return value

pth =  "/project/ukbblatent/soumick/corr_analyses/F20208v2/v0.9.6_sixth_basket/summaries"

methods_merge = [
        "MCCA_mergeWithAvg", "align_MCCA_mergeWithAvg", "align_MCCA_mergeWithPCA",
        "pairwiseCCA", "align_pairwiseCCA", "PCA"
    ]
methods_merge.sort(key=len, reverse=True)
methods_selection = [
        "posthoc_cosine", "posthoc_rscore"
    ]
filts = [
    "thres_gt50p", "thres_gt60p", "thres_gt70p"
]
corr_att_dict = {f"{m}_{f}": pd.read_table(f"{pth.replace('summaries','shortlisted_latents_models/summaries')}/{m}_{f}/6Models_f0_5Seeds_spearmanETApb_n_correlated_attributes.tsv") for m in methods_selection for f in filts}


corr_att = pd.read_table(f"{pth}/6Models_f0_5Seeds_spearmanETApb_n_correlated_attributes.tsv")
corr_att_seed = pd.read_table(f"{pth}/6Models_f0_5Seeds_spearmanETApb_seedwise_n_correlated_attributes.tsv")
corr_att_merge = pd.read_table(f"{pth.replace('summaries','merged_latents_models/summaries')}/6Models_f0_5Seeds_spearmanETApb_n_correlated_attributes.tsv")




# For Individual Trainings
individual_trainings = get_counts(corr_att_seed, ['Model', 'Seed', 'Strength'], 'Compare', 'baseline_bodymeas')

# For Ensemble
ensemble = get_counts(corr_att, ['Model', 'Strength'], 'Compare', 'baseline_bodymeas')

# For Latent Selection
latent_selection = {
    key: get_counts(df, ['Model', 'Strength'], 'Compare', 'baseline_bodymeas') for key, df in corr_att_dict.items()
}
latent_selection = pd.concat(latent_selection)

# For Latent Merging
latent_merging = corr_att_merge[corr_att_merge.Strength.isin(["Strong", "Moderate"])]
split_data = corr_att_merge['Model'].apply(lambda x: pd.Series(split_method_model(x, methods_merge), index=['Method', 'Model']))
latent_merging = latent_merging.drop(columns=['Model'])
latent_merging = latent_merging.join(split_data)
latent_merging_grouped = latent_merging.groupby(['Method', 'Model', 'Strength'])['UniqueAttributeCount'].sum().unstack(fill_value=0)
#replace all occurance of UVAE_3Pheno2L_Do50 to ssVAE
latent_merging_grouped = latent_merging_grouped.reset_index()
latent_merging_grouped.Model = latent_merging_grouped.Model.str.replace("UVAE_3Pheno2L_Do50", "ssVAE").str.replace("DiffAEFP16", "DiffAE")
latent_merging_grouped = latent_merging_grouped.set_index(['Method', 'Model'])







models = ['VAE', 'Factor VAE', 'ssVAE', 'IDVAE', 'CIiVAE', 'DiffAE']
strengths = ['Moderate', 'Strong']

# Header
header = r"""
\begin{landscape}
\begin{table}[]
\centering
\begin{tabular}{cc|cccccccccccc}
\hline
 &  & \multicolumn{2}{c}{VAE} & \multicolumn{2}{c}{Factor VAE} & \multicolumn{2}{c}{ssVAE} & \multicolumn{2}{c}{IDVAE} & \multicolumn{2}{c}{CIiVAE} & \multicolumn{2}{c}{DiffAE} \\ \hline
 &  & Moderate & Strong & Moderate & Strong & Moderate & Strong & Moderate & Strong & Moderate & Strong & Moderate & Strong \\ \hline
"""

# Individual Trainings
individual_trainings_section = r"\multirow{5}{*}{\begin{tabular}[c]{@{}c@{}}Individual\\ Trainings\end{tabular}}"
for seed in individual_trainings.index.get_level_values('Seed').unique():
    individual_trainings_section += f" & {seed} "
    for model in models:
        for strength in strengths:
            value = get_value(individual_trainings, seed, model, strength)
            individual_trainings_section += f"& {value} "
    individual_trainings_section += r"\\ \hline" + '\n'

# Ensemble
ensemble_section = r"\multicolumn{2}{c|}{Ensemble} "
for model in models:
    for strength in strengths:
        value = get_value(ensemble, model, '', strength)  # assuming ensemble df doesn't have a seed level
        ensemble_section += f"& {value} "
ensemble_section += r"\\ \hline" + '\n'

# # Latent Selection
# latent_selection_section = r"\multirow{6}{*}{\begin{tabular}[c]{@{}c@{}}Latent\\ Selection\end{tabular}}"
# for threshold in latent_selection.index.get_level_values('Threshold').unique():  # assuming 'Threshold' is the name of the level
#     latent_selection_section += f" & {threshold} "
#     for model in models:
#         for strength in strengths:
#             value = get_value(latent_selection, threshold, model, strength)
#             latent_selection_section += f"& {value} "
#     latent_selection_section += r"\\ \hline" + '\n'
latent_selection_section = ""

# Latent Merging
latent_merging_section = r"\multirow{6}{*}{\begin{tabular}[c]{@{}c@{}}Latent \\ Merging\end{tabular}}"
for method in latent_merging_grouped.index.get_level_values('Method').unique():
    latent_merging_section += f" & {method} "
    for model in models:
        for strength in strengths:
            value = get_value(latent_merging_grouped, method, model, strength)
            latent_merging_section += f"& {value} "
    latent_merging_section += r"\\ \hline" + '\n'

# Footer
footer = r"""
\end{tabular}
\end{table}
\end{landscape}
"""

# Combine all sections
latex_table = header + individual_trainings_section + ensemble_section + latent_selection_section + latent_merging_section + footer

with open('table.tex', 'w') as file:
    file.write(latex_table)