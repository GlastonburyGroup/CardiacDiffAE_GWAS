import argparse
import sys
import os
import pandas as pd

def setup_pandas_compatibility():
    """
    Set up compatibility aliases for deprecated pandas classes.
    This allows loading pickle files from older pandas versions.
    """
    # Handle numeric index module
    sys.modules['pandas.core.indexes.numeric'] = (
        pd.core.indexes.numeric if hasattr(pd.core.indexes, 'numeric') 
        else pd.core.indexes.base
    )
    
    # Handle deprecated index types
    deprecated_indices = ['Int64Index', 'Float64Index', 'UInt64Index']
    for idx_name in deprecated_indices:
        if not hasattr(pd, idx_name):
            setattr(pd, idx_name, pd.Index)
        if hasattr(sys.modules['pandas.core.indexes.numeric'], '__dict__'):
            setattr(sys.modules['pandas.core.indexes.numeric'], idx_name, pd.Index)
    
    # Handle deprecated categorical index
    if not hasattr(pd, 'CategoricalIndex'):
        pd.CategoricalIndex = pd.CategoricalIndex if hasattr(pd, 'CategoricalIndex') else pd.Index
    
    print("Pandas compatibility layer initialised successfully")

# Automatically set up compatibility when this module is imported
setup_pandas_compatibility()

import pickle
import numpy as np
from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def merge_codedesc(df, codedesc, code):
    df = df.merge(codedesc, left_on="DisCode", right_index=True, how="left")
    if code in ["PhecodeCh", "PhecodeLvl1"]:
        nan_rows = df[df.PhecodeString.isna()] #there will be rows without matcing Phecode due to the granularity of the code
        codedesc['PhecodeLvl1'] = codedesc.index.str[:8]
        codedesc['PhecodeCh'] = codedesc.index.str[:6]
        temp_merge = nan_rows.merge(
            codedesc,
            left_on="DisCode",
            right_on="PhecodeLvl1" if code == "PhecodeLvl1" else "PhecodeCh",
            suffixes=("_2del", ""),
            how="left"
        )
        if "key" in temp_merge:
            temp_merge = temp_merge.sort_values(by=['key', 'DisCode', "PhecodeLvl1" if code == "PhecodeLvl1" else "PhecodeCh"]).drop_duplicates(subset=['key', 'DisCode'], keep='first')
        else:
            temp_merge = temp_merge.sort_values(by=['Classifier', 'Metric', 'DisCode', "PhecodeLvl1" if code == "PhecodeLvl1" else "PhecodeCh"]).drop_duplicates(subset=['Classifier', 'Metric', 'DisCode'], keep='first')
        temp_merge = temp_merge[[col for col in temp_merge.columns if ("_2del" not in col) and (col not in ["PhecodeLvl1", "PhecodeCh"])]]
        del codedesc["PhecodeLvl1"], codedesc["PhecodeCh"]
        df = pd.concat([df[~df.PhecodeString.isna()], temp_merge])
    return df

def displots(df, path):
    #only keep F1 scores, and those with max > 0.5
    df_class_f1 = df[(df.Metric == "f1-score") & (df.ScoreMax > 0.5)]
    
    if "PhecodeString" in df_class_f1.columns:
        # Keep only the best F1 Score for each unique PhecodeString
        df_unique = df_class_f1.loc[df_class_f1.groupby("PhecodeString")["ScoreMax"].idxmax()]
        # Get the top 25 unique highest-scoring Phecodes
        df_top_phecodes = df_unique.nlargest(50, "ScoreMax")

        # Shorten PhecodeString names for display
        df_top_phecodes["ShortPhecodeString"] = df_top_phecodes["PhecodeString"].str[:50]  # Limit to 50 chars

        # Create figure with 3 subplots in 16:9 ratio
        fig, axes = plt.subplots(1, 3, figsize=(20, 9), gridspec_kw={'width_ratios': [2, 1.2, 1.5]})
    else:
        fig, ax = plt.subplots(figsize=(12, 9))
        axes = [ax]

    # --- Histogram + KDE for F1 Score Distribution ---
    sns.histplot(df_class_f1["ScoreMax"], bins=50, kde=True, color="royalblue", alpha=0.7, ax=axes[0])
    axes[0].axvline(df_class_f1["ScoreMax"].median(), color="red", linestyle="dashed", linewidth=2, label="Median")
    axes[0].set_xlabel("F1 Score (Max over folds)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Distribution of F1 Scores", fontsize=14, fontweight="bold")
    axes[0].legend()

    if "PhecodeString" in df_class_f1.columns:
        # --- Text List of Top 50 Unique Phecodes (Middle Panel) ---
        axes[1].axis("off")  # Hide axes
        top_phecode_text = "\n".join([f"{row.ShortPhecodeString:<52} {row.ScoreMax:.3f}" for _, row in df_top_phecodes.iterrows()])
        axes[1].text(-0.25, 1, top_phecode_text, fontsize=10, ha="left", va="top", family="monospace", transform=axes[1].transAxes)
        axes[1].set_title("Top 50 Unique Phecodes by F1 Score", fontsize=14, fontweight="bold")

        # --- Count Plot of Phecode Categories (Right Panel) ---
        sns.countplot(y=df_class_f1["PhecodeCategory"], 
                    order=df_class_f1["PhecodeCategory"].value_counts().index, 
                    palette="viridis", ax=axes[2])
        axes[2].set_xlabel("Count", fontsize=12)
        axes[2].set_ylabel("Phecode Category", fontsize=12)
        axes[2].set_title("Distribution of Phecode Categories", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")

def volcano_assoc(assocDFs, pth, significance_threshold = 0.05, n_correct=None):
    assocDFs["-log10p"] = -np.log10(assocDFs["p-value"])    
    assocDFs["significant"] = assocDFs["p-value"] < significance_threshold

    # Extract the first half of DisCode, to get the category
    assocDFs.loc[assocDFs["significant"], "DisCat"] = assocDFs.loc[assocDFs["significant"], "DisCode"].apply(lambda x: x.split("_")[0])

    # Assign unique colours to each unique DisCat
    unique_groups = assocDFs.loc[assocDFs["significant"], "DisCat"].unique()
    palette = sns.color_palette("husl", len(unique_groups))  # Generate distinct colours
    colour_map = dict(zip(unique_groups, palette))  # Map each unique DisCat to a colour

    # Default colour for non-significant points
    assocDFs["plot_colour"] = "grey"

    # Assign colours based on DisCat
    assocDFs.loc[assocDFs["significant"], "plot_colour"] = assocDFs.loc[assocDFs["significant"], "DisCat"].map(colour_map)

    # Plot the volcano plot
    plt.figure(figsize=(10, 6))
    plt.scatter(assocDFs["EffectSize"], assocDFs["-log10p"], 
                c=assocDFs["plot_colour"], alpha=0.75, edgecolors="black")

    # # Highlight the most significant associations
    # top_hits = assocDFs.nsmallest(5, "p-value")  # 5 most significant
    # for i, row in top_hits.iterrows():
    #     plt.text(row["EffectSize"], row["-log10p"], row["DisCode"], 
    #              fontsize=9, ha='right', color='black')

    # Add reference lines for significance threshold
    plt.axhline(-np.log10(significance_threshold), linestyle="--", color="blue", label=f'p = {significance_threshold}')
    if n_correct:
        plt.axhline(-np.log10(significance_threshold/n_correct), linestyle="--", color="red", label=f'p = {significance_threshold/n_correct}')

    legend_elements = [Patch(facecolor="grey", edgecolor="black", label="Not Significant")]  # Grey for non-significant points
    legend_elements += [Patch(facecolor=colour_map[group], edgecolor="black", label=group) for group in unique_groups]

    plt.legend(handles=legend_elements, title="Disease Groups", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.xlabel("Effect Size")
    plt.ylabel("-log10(p-value)")
    plt.title("Volcano Plot of Associations")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(pth)

def process_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out_root', action="store", default="/project/ukbblatent/soumick/ML_analyses/Atlas/MultiOrganV3_ModelSelect_204Liver")
    parser.add_argument('--codes', action="store", default="PhecodeLvl1")
    # parser.add_argument('--codes', action="store", default="Phecode,PhecodeCh,PhecodeLvl1")
    parser.add_argument('--modes', action="store", default="diag,diag10y,diag5y,prog5y,prog")
    # parser.add_argument('--modes', action="store", default="diag10y")
    parser.add_argument('--model_tag', action="store", default="DiffAE_F20204v3_Liver_Mag.08")
    parser.add_argument('--organ', action="store", default="Liver")
    
    parser.add_argument('--code_desc', action="store", default="/project/ukbblatent/clinicaldata/Phecode_mapX_filtered.csv", help="Path to the file containing the code descriptions. Currently works only for Phecodes. Leave it blank if not desired.")

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def main():
    args, unknown_args = process_arguments()

    args.codes = args.codes.split(",")
    args.modes = args.modes.split(",")
    
    if bool(args.code_desc):
        codedesc = pd.read_csv(args.code_desc)
        codedesc = codedesc[['Phecode', 'PhecodeString', 'PhecodeCategory']].drop_duplicates().set_index("Phecode")

    for code in args.codes:
        for mode in args.modes:
            assocDFs = []
            classiDFs = []

            pkl_pths = glob(f"{args.out_root}/{args.organ}/{code}/{mode}/{args.model_tag}/{args.model_tag}_*_results.pkl")
            print(f"Processing {len(pkl_pths)} files for {code} in {mode} mode.")
            for p in tqdm(pkl_pths):            
                with open(p, "rb") as f:
                    data = pickle.load(f)

                    #Prepare metadata
                    path_parts = p.split(os.path.sep)
                    discode = path_parts[-1].replace(f"{args.model_tag}_", "").replace("_results.pkl", "")
                    code = p.split(os.path.sep)[-4]
                    mode = p.split(os.path.sep)[-3]    
                    
                    #Process associations
                    df = pd.DataFrame([{**data['BinCAT_Disease_Assoc_dataset'][key], 'key': key} for key in data['BinCAT_Disease_Assoc_dataset'].keys()])
                    if len(df) > 0:
                        df = df[['TrainSize', 'EffectSize', 'StdError', 'p-value', 'key']]
                        df['DisCode'] = discode
                        df['CodeType'] = code
                        df['Mode'] = mode
                        assocDFs.append(df)

                    #Process classification results
                    classifiers = [k for k in data.keys() if "_Assoc_" not in k]
                    class_res = []
                    for clsf in classifiers:
                        for fold in data[clsf].keys():
                            try:
                                res = data[clsf][fold]['ClassifRprt_TestSet'].loc['weighted avg']
                                for metric in res.keys():
                                    class_res.append({
                                        "Classifier": clsf,
                                        "Fold": fold,
                                        "Metric": metric,
                                        "Score": res[metric],
                                        "DisCode": discode,
                                        "CodeType": code,
                                        "Mode": mode
                                    })
                            except:
                                pass
                    if len(class_res) > 0:
                        class_res = pd.DataFrame(class_res)
                        classiDFs.append(class_res)

            summary = ""
            
            if len(assocDFs) > 0:
                assocDFs = pd.concat(assocDFs)
                if bool(args.code_desc):
                    assocDFs = merge_codedesc(assocDFs, codedesc, code)
                assocDFs['p-value'] = assocDFs['p-value'].astype(float)
                
                if len(assocDFs[assocDFs['p-value'] < 0.05]) > 0:
                    assocDFs[assocDFs['p-value'] < 0.05].to_csv(f"{args.out_root}/{args.organ}/{code}/{mode}/{args.model_tag}/SigAssociations.csv", index=False)
                    volcano_assoc(assocDFs, significance_threshold = 0.05, n_correct=None, pth=f"{args.out_root}/{args.organ}/{code}/{mode}/{args.model_tag}/VolcanoPlot_Associations.png")
                    summary += f"Among {assocDFs.DisCode.nunique()} disease associations tested successfully, {assocDFs[assocDFs['p-value'] < 0.05].DisCode.nunique()} were significant at p < 0.05.\n"
                    summary += f"Significant association codes: {', '.join(sorted(assocDFs[assocDFs['p-value'] < 0.05].DisCode.unique()))}\n\n"
                    if bool(args.code_desc):
                        summary += f"Significant association names: {', '.join(sorted(assocDFs[assocDFs['p-value'] < 0.05].PhecodeString.unique()))}\n\n"
                        summary += f"Significant association categories with counts:\n{assocDFs[assocDFs['p-value'] < 0.05].PhecodeCategory.value_counts()}\n\n"    

            if len(classiDFs) > 0:
                classiDFs = pd.concat(classiDFs)
                consolidated = classiDFs.groupby(["Classifier", "Metric", "DisCode", "CodeType", "Mode"]).agg(
                                    ScoreMean=("Score", "mean"),
                                    ScoreStd=("Score", "std"),
                                    ScoreMedian=("Score", "median"),
                                    ScoreIQR=("Score", lambda x: x.quantile(0.75) - x.quantile(0.25)),
                                    ScoreMax=("Score", "max"),
                                ).reset_index()
                if bool(args.code_desc):
                    consolidated = merge_codedesc(consolidated, codedesc, code)
                consolidated.to_csv(f"{args.out_root}/{args.organ}/{code}/{mode}/{args.model_tag}/ClassifResults_Consolidated.csv", index=False)
                displots(consolidated, f"{args.out_root}/{args.organ}/{code}/{mode}/{args.model_tag}/SuccClassifResults_Distribution.png")
                classify_succsess = consolidated[(consolidated.Metric == "f1-score") & (consolidated.ScoreMax > 0.5)]
                summary += f"{classify_succsess.DisCode.nunique()} diseases were classified with max (across folds) f1-score > 0.5.\n"
                summary += f"Successfully classified disease codes: {', '.join(sorted(classify_succsess.DisCode.unique()))}\n\n"
                if bool(args.code_desc):
                    summary += f"Successfully classified disease names: {', '.join(sorted(classify_succsess.PhecodeString.unique()))}\n\n"
                    summary += f"Successfully classified disease categories with counts:\n{classify_succsess.PhecodeCategory.value_counts()}\n\n"

            if (len(assocDFs) > 0 and len(assocDFs[assocDFs['p-value'] < 0.05]) > 0) and len(classiDFs) > 0:
                displots(classify_succsess[classify_succsess.DisCode.isin(assocDFs[assocDFs['p-value'] < 0.05].DisCode.unique())], f"{args.out_root}/{args.organ}/{code}/{mode}/{args.model_tag}/SigSuccClassifResults_Distribution.png")
                summary += f"Successfully classified disease codes with significant association: {', '.join(sorted(list(set(assocDFs[assocDFs['p-value'] < 0.05].DisCode.unique()).intersection(classify_succsess.DisCode.unique()))))}\n\n"
                if bool(args.code_desc):
                    summary += f"Successfully classified disease names with significant association: {', '.join(sorted(list(set(assocDFs[assocDFs['p-value'] < 0.05].PhecodeString.unique()).intersection(classify_succsess.PhecodeString.unique()))))}\n\n"
                    summary += f"Successfully classified disease categories with significant association:\n{classify_succsess[classify_succsess.DisCode.isin(assocDFs[assocDFs['p-value'] < 0.05].DisCode.unique())].PhecodeCategory.value_counts()}\n\n"

            if summary:
                with open(f"{args.out_root}/{args.organ}/{code}/{mode}/{args.model_tag}/Summary.txt", "w") as f:
                    f.write(summary)


if __name__ == "__main__":
    main()