import argparse
import sys
import os
import pandas as pd

# Add compatibility layer for older pandas pickle files
sys.modules['pandas.core.indexes.numeric'] = pd.core.indexes.numeric if hasattr(pd.core.indexes, 'numeric') else pd.core.indexes.base

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def phenoplots(df, path):
    #only keep F1 scores, and those with max > 0.5
    df_class_f1 = df[(df.Metric == "R-squared") & (df.ScoreMax > 0.5)]
    
    df_unique = df_class_f1.loc[df_class_f1.groupby("Pheno")["ScoreMax"].idxmax()]

    # Get the top 25 unique highest-scoring Phenos
    df_top_phenos = df_unique.nlargest(50, "ScoreMax")

    # Shorten PhenoString names for display
    df_top_phenos["ShortPhenoString"] = df_top_phenos["Pheno"].str[:50]  # Limit to 50 chars

    # Create figure with 3 subplots in 16:9 ratio
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [2, 1.2]})

    # --- Histogram + KDE for F1 Score Distribution ---
    sns.histplot(df_class_f1["ScoreMax"], bins=50, kde=True, color="royalblue", alpha=0.7, ax=axes[0])
    axes[0].axvline(df_class_f1["ScoreMax"].median(), color="red", linestyle="dashed", linewidth=2, label="Median")
    axes[0].set_xlabel("R-squared Score (Max over folds)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Distribution of R-squared Scores", fontsize=14, fontweight="bold")
    axes[0].legend()
    
    # --- Text List of Top 50 Unique Phecodes (Middle Panel) ---
    axes[1].axis("off")  # Hide axes
    top_pheno_text = "\n".join([f"{row.ShortPhenoString:<52} {row.ScoreMax:.3f}" for _, row in df_top_phenos.iterrows()])
    axes[1].text(0, 1, top_pheno_text, fontsize=10, ha="left", va="top", family="monospace", transform=axes[1].transAxes)
    axes[1].set_title("Top 50 Unique Phenos by R-sqaured Score", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")

def volcano_assoc(assocDFs, pth, significance_threshold = 0.05, n_correct=None):
    assocDFs["-log10p"] = -np.log10(assocDFs["p-value"])    
    assocDFs["significant"] = assocDFs["p-value"] < significance_threshold

    # Assign unique colours to each unique Pheno
    unique_groups = assocDFs.loc[assocDFs["significant"], "Pheno"].unique()
    palette = sns.color_palette("husl", len(unique_groups))  # Generate distinct colours
    colour_map = dict(zip(unique_groups, palette))  # Map each unique pheno to a colour

    # Default colour for non-significant points
    assocDFs["plot_colour"] = "grey"

    # Assign colours based on Pheno
    assocDFs.loc[assocDFs["significant"], "plot_colour"] = assocDFs.loc[assocDFs["significant"], "Pheno"].map(colour_map)

    # Plot the volcano plot
    plt.figure(figsize=(10, 6))
    plt.scatter(assocDFs["EffectSize"], assocDFs["-log10p"], 
                c=assocDFs["plot_colour"], alpha=0.75, edgecolors="black")

    # # Highlight the most significant associations
    # top_hits = assocDFs.nsmallest(5, "p-value")  # 5 most significant
    # for i, row in top_hits.iterrows():
    #     plt.text(row["EffectSize"], row["-log10p"], row["Pheno"], 
    #              fontsize=9, ha='right', color='black')

    # Add reference lines for significance threshold
    plt.axhline(-np.log10(significance_threshold), linestyle="--", color="blue", label=f'p = {significance_threshold}')
    if n_correct:
        plt.axhline(-np.log10(significance_threshold/n_correct), linestyle="--", color="red", label=f'p = {significance_threshold/n_correct}')

    legend_elements = [Patch(facecolor="grey", edgecolor="black", label="Not Significant")]  # Grey for non-significant points
    legend_elements += [Patch(facecolor=colour_map[group], edgecolor="black", label=group) for group in unique_groups]

    plt.legend(handles=legend_elements, title="Pheno", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.xlabel("Effect Size")
    plt.ylabel("-log10(p-value)")
    plt.title("Volcano Plot of Associations")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(pth)

def process_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out_root', action="store", default="/project/ukbblatent/soumick/ML_analyses/Atlas/MultiOrganV3_Initial")
    parser.add_argument('--model_tag', action="store", default="DiffAE_F20259v3_Pancreas_FITPARAMS")
    parser.add_argument('--organ', action="store", default="Pancreas")

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def main():
    args, unknown_args = process_arguments()
    
    pkl_pths = glob(f"{args.out_root}/{args.organ}/*/{args.model_tag}/{args.model_tag}_*_results.pkl")
    
    assocDFs = []
    regressDFs = []
            
    for p in tqdm(pkl_pths):            
        with open(p, "rb") as f:
            data = pickle.load(f)

            #Prepare metadata
            path_parts = p.split(os.path.sep)
            pheno_file = path_parts[-1].replace(f"{args.model_tag}_", "").replace("_results.pkl", "")
            pheno_type = path_parts[-3] 
            phenos = list(set([k.replace("_Assoc_dataset", "") for k in data.keys() if k.endswith("_Assoc_dataset")]))
            
            for pheno in phenos:             
            
                #Process associations
                df = pd.DataFrame([{**data[f'{pheno}_Assoc_dataset'][key], 'key': key} for key in data[f'{pheno}_Assoc_dataset'].keys()])
                if len(df) > 0:
                    df = df[['TrainSize', 'EffectSize', 'StdError', 'p-value', 'key']]
                    df['Pheno'] = pheno
                    df['PhenoType'] = pheno_type
                    df['PhenoFile'] = pheno_file
                    assocDFs.append(df)

                #Process regressfication results
                regressors = [k for k in data.keys() if k.startswith(pheno) and "_Assoc_" not in k]
                class_res = []
                for clsf in regressors:
                    for fold in data[clsf].keys():
                        try:
                            class_res.append({
                                    "Regressor": clsf,
                                    "Fold": fold,
                                    "Metric": "R-squared",
                                    "Score": data[clsf][fold]['R-squared_TestSet'],
                                    'Pheno': pheno,
                                    'PhenoType': pheno_type,
                                    'PhenoFile': pheno_file,
                                })
                            class_res.append({
                                    "Regressor": clsf,
                                    "Fold": fold,
                                    "Metric": "MSE",
                                    "Score": data[clsf][fold]['MSE_TestSet'],
                                    'Pheno': pheno,
                                    'PhenoType': pheno_type,
                                    'PhenoFile': pheno_file,
                                })
                        except:
                            pass
                if len(class_res) > 0:
                    class_res = pd.DataFrame(class_res)
                    regressDFs.append(class_res)

    summary = ""
    
    if len(assocDFs) > 0:
        assocDFs = pd.concat(assocDFs)
        assocDFs['p-value'] = assocDFs['p-value'].astype(float)
        
        if len(assocDFs[assocDFs['p-value'] < 0.05]) > 0:
            assocDFs[assocDFs['p-value'] < 0.05].to_csv(f"{os.path.dirname(p)}/SigAssociations.csv", index=False)
            volcano_assoc(assocDFs, significance_threshold = 0.05, n_correct=None, pth=f"{os.path.dirname(p)}/VolcanoPlot_Associations.png")
            summary += f"Among {assocDFs.Pheno.nunique()} pheno associations tested successfully, {assocDFs[assocDFs['p-value'] < 0.05].Pheno.nunique()} were significant at p < 0.05.\n"
            summary += f"Significant associations: {', '.join(sorted(assocDFs[assocDFs['p-value'] < 0.05].Pheno.unique()))}\n\n"
            
    if len(regressDFs) > 0:
        regressDFs = pd.concat(regressDFs)
        consolidated = regressDFs.groupby(["Regressor", "Metric", "Pheno", "PhenoType", "PhenoFile"]).agg(
                            ScoreMean=("Score", "mean"),
                            ScoreStd=("Score", "std"),
                            ScoreMedian=("Score", "median"),
                            ScoreIQR=("Score", lambda x: x.quantile(0.75) - x.quantile(0.25)),
                            ScoreMax=("Score", "max"),
                        ).reset_index()
        consolidated.to_csv(f"{os.path.dirname(p)}/RegressfResults_Consolidated.csv", index=False)
        phenoplots(consolidated, f"{os.path.dirname(p)}/SuccRegressfResults_Distribution.png")
        regressfy_succsess = consolidated[(consolidated.Metric == "R-squared") & (consolidated.ScoreMax > 0.5)]
        summary += f"{regressfy_succsess.Pheno.nunique()} phenos were regressed with max (across folds) R-squared > 0.5.\n"
        summary += f"Successfully regressed phenos: {', '.join(sorted(regressfy_succsess.Pheno.unique()))}\n\n"

    if (len(assocDFs) > 0 and len(assocDFs[assocDFs['p-value'] < 0.05]) > 0) and len(regressDFs) > 0:
        phenoplots(regressfy_succsess[regressfy_succsess.Pheno.isin(assocDFs[assocDFs['p-value'] < 0.05].Pheno.unique())], f"{os.path.dirname(p)}/SigSuccRegressfResults_Distribution.png")
        summary += f"Successfully regressed phenos with significant association: {', '.join(sorted(list(set(assocDFs[assocDFs['p-value'] < 0.05].Pheno.unique()).intersection(regressfy_succsess.Pheno.unique()))))}\n\n"
        
    if summary:
        with open(f"{os.path.dirname(p)}/Summary.txt", "w") as f:
            f.write(summary)


if __name__ == "__main__":
    main()