# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import pandas as pd
from glob import glob
from tqdm import tqdm
import pickle
import sys
import itertools
import argparse

# sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/group/glastonbury/soumick/MyCodes/GitLab/tricorder")

import numpy as np
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.power import tt_ind_solve_power

import pyreadr
from rds2py import read_rds

import matplotlib.pyplot as plt
import seaborn as sns

from utils.stats.delong import delong_roc_test
from utils.stats.power import simulate_power_wilcoxon
from utils.python_utils import recursive_defaultdict

sns.set_style("whitegrid")

# for the prevalence plots, defination of the percentiles
percentiles = [5, 10, 20, 80, 90, 95]
x_labels = ['Bottom 5%', 'Bottom 10%', 'Bottom 20%', 'Top 20%', 'Top 10%', 'Top 5%']

# for the prevalence plots, defination of the colours and markers
colours = ['salmon', 'turquoise', 'mediumpurple', 'sandybrown', 'lightgreen', 'palevioletred']
markers = ['o', 's', 'D', '^', 'v', 'P']

# %% command-line arguments

def getARGSParser():
    parser = argparse.ArgumentParser(description='MultiPRS Script')
    parser.add_argument('--output_root', type=str, help='Path to the multi-PRS analyses root directory', default="/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/analyses/panCohort_auto_lw_gw_20PC_1000it_caucasiancohort_king0p0625")
    parser.add_argument('--disease_root', type=str, help='path to the root directory where the disease cohort fils are storred', default="/project/ukbblatent/clinicaldata/binary_disease_cohorts/F20208v3_nonDiscov/caucasian_king0p0625")

    parser.add_argument('--pth_prs_prefix', type=str, help='Prefix (full path) to the PRS file', default="/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/run_ext_basic_lw_gw_indep_FiltMAF_")
    parser.add_argument('--rds_pres_suffix', type=str, help='Suffix after the pheno name in the RDS file name', default=".fullDS.auto.mod.LDPred2.rds")
    parser.add_argument('--rds_tag_prs', type=str, help='tag PRS present in the rds file name', default="auto.mod")
    parser.add_argument('--tag_data', type=str, help='tag PRS model', default="resNdata.basic")
    parser.add_argument('--tag_prs', type=str, help='tag PRS inside the rds file', default="pred_auto")

    parser.add_argument('--is_pancohort', action=argparse.BooleanOptionalAction, default=True, help='Whether to run/plot the pancohort probability predictions')
    parser.add_argument('--save_plots', action=argparse.BooleanOptionalAction, default=True, help='Whether to save the plots')
    parser.add_argument('--sex_stratified', action=argparse.BooleanOptionalAction, default=True, help='Whether to perform in sex-stratified manner as well')

    parser.add_argument('--obtain_summary', action=argparse.BooleanOptionalAction, default=True, help='Whether to obtain summary scores and perform statistical stats. If set to False, it will be presumed that they are already performed and the required files are present')
    parser.add_argument('--obtain_pairwise_improvements', action=argparse.BooleanOptionalAction, default=False, help='Whether to print the pairwise improvements')
    
    parser.add_argument('--plot_box_AUC', action=argparse.BooleanOptionalAction, default=False, help='Whether to plot the box plots for AUC')
    parser.add_argument('--plot_box_logOR', action=argparse.BooleanOptionalAction, default=False, help='Whether to plot the box plots for logOR [If pancohort, both pancohort and disease cohort will be plotted]')
    parser.add_argument('--plot_prevalence_prob_diseasecohort', action=argparse.BooleanOptionalAction, default=False, help='Whether to plot the prevalence vs probability plots for the disease cohort')
    parser.add_argument('--plot_prevalence_prs_pancohort', action=argparse.BooleanOptionalAction, default=True, help='Whether to plot the prevalence vs PRS scores (best performing single PRS) plots for the pancohort')
    parser.add_argument('--plot_prevalence_prob_pancohort', action=argparse.BooleanOptionalAction, default=False, help='Whether to plot the prevalence vs probability plots for the pancohort')

    return parser

# %% Support functions for the processing of the raw results and compute the stats

def get_stats(score_m0, score_m1):
    t_stat, p_value_t_test = stats.ttest_rel(score_m0, score_m1)
    print("Paired t-test - p-value:", p_value_t_test, f"({'Sig' if p_value_t_test < 0.05 else 'Not Sig'})")

    effect_size = (np.mean(score_m0-score_m1) / np.std(score_m0-score_m1))

    power = tt_ind_solve_power(effect_size=effect_size, nobs1=5, alpha=0.05, ratio=1, alternative='two-sided')
    print("Statistical Power (Paired t-test):", power)

    stat, p_value_wilcoxon = stats.wilcoxon(score_m0, score_m1)
    print("Wilcoxon Signed-Rank Test - p-value:", p_value_wilcoxon, f"({'Sig' if p_value_wilcoxon < 0.05 else 'Not Sig'})")
    
    estimated_power = simulate_power_wilcoxon(len(score_m0), effect_size)
    print("Estimated Power:", estimated_power)

    return p_value_t_test, power, p_value_wilcoxon, estimated_power

def get_sig(method0, method1, comboDF=None, pancohort=False):
    if type(method0) == str:
        method0 = comboDF[comboDF['model'] == method0]
    if type(method1) == str:
        method1 = comboDF[comboDF['model'] == method1]

    method0 = method0.sort_values('fold')
    method1 = method1.sort_values('fold')

    print("AUC:")
    p_value_t_test_AUC, power_t_test_AUC, p_value_wilcoxon_AUC, power_wilcoxon_AUC = get_stats(method0['AUC_test'].values, method1['AUC_test'].values)

    print("logOR:")
    p_value_t_test_logOR, power_t_test_logOR, p_value_wilcoxon_logOR, power_wilcoxon_logOR = get_stats(method0['logOR_test'].values, method1['logOR_test'].values)

    if pancohort:
        print("logOR (pancohort):")
        p_value_t_test_logOR_pancohort, power_t_test_logOR_pancohort, p_value_wilcoxon_logOR_pancohort, power_wilcoxon_logOR_pancohort = get_stats(method0['logOR_test_pancohort'].values, method1['logOR_test_pancohort'].values)

        return p_value_t_test_AUC, power_t_test_AUC, p_value_wilcoxon_AUC, power_wilcoxon_AUC, p_value_t_test_logOR, power_t_test_logOR, p_value_wilcoxon_logOR, power_wilcoxon_logOR, p_value_t_test_logOR_pancohort, power_t_test_logOR_pancohort, p_value_wilcoxon_logOR_pancohort, power_wilcoxon_logOR_pancohort
    else:
        return p_value_t_test_AUC, power_t_test_AUC, p_value_wilcoxon_AUC, power_wilcoxon_AUC, p_value_t_test_logOR, power_t_test_logOR, p_value_wilcoxon_logOR, power_wilcoxon_logOR

def processSingleMax(df, tag):
    df = df.iloc[df['AUC_test'].argmax()]
    print(df.name)

    df = pd.DataFrame(df['pred_probs_test'])
    return df.rename({"predicted": tag}, axis=1)

def processRes(df, tag):
    df = pd.DataFrame(df['pred_probs_test'])
    return df.rename({"predicted": tag}, axis=1)

def getSummaryDF(df, pancohort=False):
    if pancohort:
        cols = ['AUC_test', 'logOR_test', 'logOR_test_pancohort']
    else:
        cols = ['AUC_test', 'logOR_test']
    median_df = df.groupby('tag')[cols].median()   
    iqr_df = (df.groupby('tag')[cols].quantile(0.75) - df.groupby('tag')[cols].quantile(0.25))
    merged_df = median_df.join(iqr_df, lsuffix='_median', rsuffix='_iqr')
    merged_df['AUC_test'] = merged_df.apply(lambda row: f"{row['AUC_test_median']:.4f} ± {row['AUC_test_iqr']:.4f}", axis=1)
    merged_df['logOR_test'] = merged_df.apply(lambda row: f"{row['logOR_test_median']:.4f} ± {row['logOR_test_iqr']:.4f}", axis=1)
    if pancohort:
        merged_df['logOR_test_pancohort'] = merged_df.apply(lambda row: f"{row['logOR_test_pancohort_median']:.4f} ± {row['logOR_test_pancohort_iqr']:.4f}", axis=1)
    return merged_df[cols].reset_index()

def compareSummaryPairs(df, all_combinations, disease, pancohort=False):
    collect = []
    for combination in all_combinations:
        comparison1, comparison2 = combination
        print(f"{comparison1[2]} [vs] {comparison2[2]}-----")

        df_method1 = df[df.apply(lambda row: (row['method'], row['res_type']) == (comparison1[0], comparison1[1]), axis=1)]
        df_method2 = df[df.apply(lambda row: (row['method'], row['res_type']) == (comparison2[0], comparison2[1]), axis=1)]
        
        if pancohort:
            p_value_t_test_AUC, power_t_test_AUC, p_value_wilcoxon_AUC, power_wilcoxon_AUC, p_value_t_test_logOR, power_t_test_logOR, p_value_wilcoxon_logOR, power_wilcoxon_logOR, p_value_t_test_logOR_pancohort, power_t_test_logOR_pancohort, p_value_wilcoxon_logOR_pancohort, power_wilcoxon_logOR_pancohort = get_sig(df_method1, df_method2, pancohort=True)
        else:
            p_value_t_test_AUC, power_t_test_AUC, p_value_wilcoxon_AUC, power_wilcoxon_AUC, p_value_t_test_logOR, power_t_test_logOR, p_value_wilcoxon_logOR, power_wilcoxon_logOR = get_sig(df_method1, df_method2)

        res = {
                "Disease": disease,
                "Comparison": f"{comparison1[2]} [vs] {comparison2[2]}",
                "p_value_t_test_AUC": p_value_t_test_AUC,
                "power_t_test_AUC": power_t_test_AUC,
                "p_value_wilcoxon_AUC": p_value_wilcoxon_AUC,
                "power_wilcoxon_AUC": power_wilcoxon_AUC,
                "p_value_t_test_logOR": p_value_t_test_logOR,
                "power_t_test_logOR": power_t_test_logOR,
                "p_value_wilcoxon_logOR": p_value_wilcoxon_logOR,
                "power_wilcoxon_logOR": power_wilcoxon_logOR
            }
        if pancohort:
            res.update({
                "p_value_t_test_logOR_pancohort": p_value_t_test_logOR_pancohort,
                "power_t_test_logOR_pancohort": power_t_test_logOR_pancohort,
                "p_value_wilcoxon_logOR_pancohort": p_value_wilcoxon_logOR_pancohort,
                "power_wilcoxon_logOR_pancohort": power_wilcoxon_logOR_pancohort
            })
        collect.append(res)
        
    return pd.DataFrame(collect)

def getIndividualProbs(res_method, res_type, get_pancohort=False):
    if "single" in res_type.lower():
        res_method = res_method.iloc[res_method['AUC_test'].argmax()]
        print(f"Selected latent: {res_method.name.replace('PRS:', '')}")
    if get_pancohort:
        return pd.DataFrame(res_method['pred_probs_test_pancohort']), res_method
    else:
        return pd.DataFrame(res_method['pred_probs_test']), res_method

def compareAUCPairs(res, all_combinations, disease, diseaseDF):
    collect = []
    for combination in all_combinations:
        comparison1, comparison2 = combination
        print(f"{comparison1[2]} [vs] {comparison2[2]}-----")
        
        collect_fold = []
        for fold in res.keys():
            pred_prob_m0, _ = getIndividualProbs(res[fold][comparison1[0]][comparison1[1]], comparison1[1])
            pred_prob_m1, _ = getIndividualProbs(res[fold][comparison2[0]][comparison2[1]], comparison2[1])
            pred_prob = pred_prob_m0.join(pred_prob_m1, lsuffix="_m0", rsuffix="_m1")
            combined = pred_prob.join(diseaseDF[['BinCAT_Disease']])
            delongP = delong_roc_test(combined['BinCAT_Disease'], combined['predicted_m0'], combined['predicted_m1'])
            print(f"DeLong's p-value [{fold}]:", delongP, f"({'Sig' if delongP < 0.05 else 'Not Sig'})")
            collect_fold.append(combined)
        collect_fold = pd.concat(collect_fold)
        delongP = delong_roc_test(collect_fold['BinCAT_Disease'], collect_fold['predicted_m0'], collect_fold['predicted_m1'])
        print(f"DeLong's p-value [Across Folds]:", delongP, f"({'Sig' if delongP < 0.05 else 'Not Sig'})")

        collect.append(
            {
                "Disease": disease,
                "Comparison": f"{comparison1[2]} [vs] {comparison2[2]}",
                "p_value_DeLong_AUC": delongP
            }
        )
    return pd.DataFrame(collect)


# %% Support functions related to the plotting

def prevalence_and_ci(pred_probs, true_labels, percentile, confidence=0.95):
    if percentile < 50:
        group = pred_probs <= np.percentile(pred_probs, percentile)
    else:
        group = pred_probs >= np.percentile(pred_probs, percentile)

    prevalence = true_labels[group].mean()
    n = len(true_labels[group])
    se = np.sqrt(prevalence * (1 - prevalence) / n)  
    ci_half_width = se * norm.ppf((1 + confidence) / 2)  
    return prevalence, prevalence - ci_half_width, prevalence + ci_half_width, n

def z_test(p_hat, p_0, n):
    se = np.sqrt(p_0 * (1 - p_0) / n)  
    z = (p_hat - p_0) / se  
    p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed test
    return p_value

def is_significant(p_hat, p_0, n, alpha=0.05):
    p_value = z_test(p_hat, p_0, n)
    return p_value < alpha

def generate_sequence_plotdelta(N):
    sequence = []
    if N == 1:
        sequence.append(0)
    else:
        for i in range(-(N//2), N//2 + 1):
            sequence.append(i * 0.1)
        if N % 2 == 0:
            sequence.remove(0)
    return sequence

def plot_prevalence(df, res_keys=['Pred_Covariates', 'Pred_PRS'], res_labels=['Covariates', 'PRS'], res_colours=['orange', 'blue'], res_fmts=['o', 's'], col_disease='BinCAT_Disease', plot_title='prova!', individual_prev_line=False, save_path=""):
    #supply Pred_Covariates, Pred_PRS and BinCAT_Disease as columns of the DataFrame 

    plotdelta = generate_sequence_plotdelta(len(res_keys))
    x_ticks = np.arange(len(x_labels))

    overall_prevalence = df[col_disease].mean()
    mid_percentile = x_ticks.mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(res_keys)):
        df_curr = df[[res_keys[i], col_disease]].dropna() #this will handle the sex-stratified case
        results = [prevalence_and_ci(df_curr[res_keys[i]], df_curr[col_disease], p) for p in percentiles]
        bottom, bottom_cis_lower, bottom_cis_upper, bottom_n = zip(*results[:3])
        top, top_cis_lower, top_cis_upper, top_n = zip(*results[3:])        
        
        ax.errorbar(np.arange(len(bottom)) + plotdelta[i], bottom, yerr=[np.subtract(bottom, bottom_cis_lower), np.subtract(bottom_cis_upper, bottom)], fmt=res_fmts[i], color=res_colours[i], label=res_labels[i], capsize=5)
        ax.errorbar(np.arange(len(top)) + 3 + plotdelta[i], top, yerr=[np.subtract(top, top_cis_lower), np.subtract(top_cis_upper, top)], fmt=res_fmts[i], color=res_colours[i], capsize=5)

        if individual_prev_line:
            ax.axhline(y=df_curr[col_disease].mean(), color=res_colours[i], linestyle='--')

    if not individual_prev_line:
        ax.axhline(y=overall_prevalence, color='gray', linestyle='--')
    ax.axvline(x=mid_percentile, color='gray', linestyle='dotted')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Prevalence')
    ax.set_title(plot_title)
    ax.legend()

    ax.grid(True)

    if bool(save_path):
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


# %% Other support functions

def getImprovedDisease(df, tag0, tag1):
        filtered_df = df[df['tag'].isin([tag0, tag1])]
        pivot_df = filtered_df.pivot(index='Disease', columns='tag', values='AUC_test').reset_index()
        result_df = pivot_df[pivot_df[tag0] < pivot_df[tag1]]
        return result_df['Disease'].tolist()

# %%
if __name__ == "__main__":
    parser = getARGSParser()
    args, _ = parser.parse_known_args()

    # %% define the interesting comparisons [TODO: make it configurable. Currently, it's a "comment-out"-based approach]
    comparisons = [
        ('GLM', 'covar', 'Covariates'),
        # ('Lasso', 'covar', 'Lasso Covariates'),
        ('GLM', 'nonPCCovar', 'non-PC Covariates'),
        # ('Lasso', 'nonPCCovar', 'Lasso non-PC Covariates'),

        # ('GLM', 'singlePRS', 'max(Single PRS)'),
        ('GLM', 'singlePRSCovar', 'max(Single PRS + Covariates)'),

        # ('Lasso', 'multiPRS', 'Multi PRS'),
        ('Lasso', 'multiPRSCovar', 'Multi PRS + Covariates'),
        ('Lasso', 'multiPRSnonPCCovar', 'Multi PRS + non-PC Covariates'),

        # ('Lasso', 'multiPRSNorm', 'Normalised Multi PRS'),
        # ('Lasso', 'multiPRSNormCovar', 'Normalised Multi PRS + Covariates'),
        # ('Lasso', 'multiPRSNormnonPCCovar', 'Normalised Multi PRS + non-PC Covariates'),
    ]

    all_combinations = list(itertools.combinations(comparisons, 2))
    interesting_combinations = {(method, res_type) for method, res_type, _ in comparisons}
    tag_lookup = {(method, res_type): tag for method, res_type, tag in comparisons}
    tag_order = [tag for _, _, tag in comparisons]

    # %% obtain the results' pickles
    results = glob(f"{args.output_root}/*_raw_results.pkl")

    # %% create summary of the scores and perform statistical tests

    if args.obtain_summary:
        collect_df = []
        collect_summary_df = []
        collect_stats = []

        with open(f"{args.output_root}/sumscores.txt", 'w') as f:
            for r in tqdm(results):
                d = os.path.basename(r).replace('_raw_results.pkl', '')
                f.write(f"\n\n{d}:-------------------------------\n")
                disease = pd.read_csv(f'{args.disease_root}/{d}.csv', low_memory=False, index_col="IID")
                disease["BinCAT_Disease"] = disease["BinCAT_Disease"].astype(int)

                df = pd.read_table(r.replace("_raw_results.pkl", "_models_test_scores.tsv"), low_memory=False)
                df['Disease'] = d.replace("-", " ")
                df['nPatients'] = len(disease) // 2

                df = df[df.apply(lambda row: (row['method'], row['res_type']) in interesting_combinations, axis=1)]
                df['tag'] = df.apply(lambda row: tag_lookup.get((row['method'], row['res_type'])), axis=1)
                df['tag'] = pd.Categorical(df['tag'], categories=tag_order, ordered=True)
                collect_df.append(df)

                summary_df = getSummaryDF(df, pancohort=args.is_pancohort)
                f.write("\nSummary scores:-")
                f.write(summary_df.to_string(index=False))
                summary_df['Disease'] = d.replace("-", " ")
                summary_df['nPatients'] = len(disease) // 2
                collect_summary_df.append(summary_df)

                summary_stats = compareSummaryPairs(df, all_combinations, d.replace("-", " "), pancohort=args.is_pancohort)

                with open(r, "rb") as pkl:
                    res = pickle.load(pkl)
                auc_delong = compareAUCPairs(res, all_combinations, d.replace("-", " "), disease)

                stats_combined = pd.merge(auc_delong, summary_stats, on=["Disease", "Comparison"])
                stats_combined['nPatients'] = len(disease) // 2
                collect_stats.append(stats_combined)

                stats2print = stats_combined[['Comparison', 'p_value_DeLong_AUC']].copy()
                stats2print['is Sig'] = stats2print['p_value_DeLong_AUC'].apply(lambda x: 'Sig' if x < 0.05 else 'Not Sig')
                f.write("\n\nDeLong's p-values:-")
                f.write(stats2print.to_string(index=False))

                f.write("\n\n-----------------------------------\n\n")

        collect_df = pd.concat(collect_df)
        collect_summary_df = pd.concat(collect_summary_df)
        collect_stats = pd.concat(collect_stats)

        collect_df.to_csv(f"{args.output_root}/combined_results.tsv", sep="\t", index=False)
        collect_summary_df.to_csv(f"{args.output_root}/combined_summary_results.tsv", sep="\t", index=False)
        collect_stats.to_csv(f"{args.output_root}/combined_stats_sig.tsv", sep="\t", index=False)

    else:
        collect_df = pd.read_table(f"{args.output_root}/combined_results.tsv", low_memory=False)
        collect_summary_df = pd.read_table(f"{args.output_root}/combined_summary_results.tsv", low_memory=False)
        collect_stats = pd.read_table(f"{args.output_root}/combined_stats_sig.tsv", low_memory=False)

    # %% print the pairwise improvements

    if args.obtain_pairwise_improvements:    
        median_df = collect_df.groupby(['Disease', 'tag'])[['AUC_test']].median().reset_index()
        
        for combination in all_combinations:
            comparison1, comparison2 = combination
            print(f"{comparison1[2]} [vs] {comparison2[2]}-----")    

            improved = getImprovedDisease(median_df, tag0=comparison1[2], tag1=comparison2[2])
            print(f"\n{comparison2[2]} resulted in higher AUC than {comparison1[2]} for {len(improved)} diseases: {', '.join(improved)}")

            improved = getImprovedDisease(median_df, tag0=comparison2[2], tag1=comparison1[2])
            print(f"\n{comparison1[2]} resulted in higher AUC than {comparison2[2]} for {len(improved)} diseases: {', '.join(improved)}")

            improvedSig = collect_stats[((collect_stats.Comparison == f'{comparison1[2]} [vs] {comparison2[2]}') | (collect_stats.Comparison == f'{comparison2[2]} [vs] {comparison1[2]}')) & (collect_stats.p_value_DeLong_AUC < 0.05)].Disease.unique()
            improvedSig = list(set(improved).intersection(set(improvedSig)))
            print(f"\n{comparison1[2]} resulted in significantly higher AUC than {comparison2[2]} for {len(improvedSig)} diseases: {', '.join(improvedSig)}")

    # %% sort the diseases based on the AUC
    sorted_diseases = collect_df[collect_df['res_type'] == 'singlePRSCovar'].groupby('Disease')['AUC_test'].mean().sort_values(ascending=False).index
    top_diseases = sorted_diseases[:(len(sorted_diseases)//2)]
    bottom_diseases = sorted_diseases[(len(sorted_diseases)//2):]

    df_top = collect_df[collect_df['Disease'].isin(top_diseases)].copy()
    df_top['Disease'] = pd.Categorical(df_top['Disease'], categories=top_diseases, ordered=True)

    df_bottom = collect_df[collect_df['Disease'].isin(bottom_diseases)].copy()
    df_bottom['Disease'] = pd.Categorical(df_bottom['Disease'], categories=bottom_diseases, ordered=True)

    # %% Create box plots for AUC_test, top N//2 diseases and bottom N//2 diseases seperately

    if args.plot_box_AUC:
        os.makedirs(os.path.join(args.output_root, "plots", "auc"), exist_ok=True)

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_top, x='Disease', y='AUC_test', hue='tag', hue_order=tag_order, palette="Set2")
        plt.title('AUC by Disease and Method', fontsize=16)
        plt.xlabel('Disease', fontsize=14)
        plt.ylabel('AUC', fontsize=14)
        plt.legend(title='Method', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        if args.save_plots:
            plt.savefig(f"{args.output_root}/plots/auc/top_{len(top_diseases)}_diseases.png", dpi=300)
        else:
            plt.show()

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_bottom, x='Disease', y='AUC_test', hue='tag', hue_order=tag_order, palette="Set2")
        plt.title('AUC by Disease and Method', fontsize=16)
        plt.xlabel('Disease', fontsize=14)
        plt.ylabel('AUC', fontsize=14)
        plt.legend(title='Method', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        if args.save_plots:
            plt.savefig(f"{args.output_root}/plots/auc/bottom_{len(bottom_diseases)}_diseases.png", dpi=300)
        else:
            plt.show()    

    # %% Create box plots for logOR_test, top N//2 diseases and bottom N//2 diseases seperately

    if args.plot_box_logOR:
        os.makedirs(os.path.join(args.output_root, "plots", "logOR"), exist_ok=True)

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_top, x='Disease', y='logOR_test', hue='tag', hue_order=tag_order, palette="Set2")
        plt.title('logOR by Disease and Method', fontsize=16)
        plt.xlabel('Disease', fontsize=14)
        plt.ylabel('logOR', fontsize=14)
        plt.legend(title='Method', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        if args.save_plots:
            plt.savefig(f"{args.output_root}/plots/logOR/top_{len(top_diseases)}_diseases.png", dpi=300)
        else:
            plt.show()

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_bottom, x='Disease', y='logOR_test', hue='tag', hue_order=tag_order, palette="Set2")
        plt.title('logOR by Disease and Method', fontsize=16)
        plt.xlabel('Disease', fontsize=14)
        plt.ylabel('logOR', fontsize=14)
        plt.legend(title='Method', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        if args.save_plots:
            plt.savefig(f"{args.output_root}/plots/logOR/bottom_{len(bottom_diseases)}_diseases.png", dpi=300)
        else:
            plt.show()

    # %% Create box plots for logOR_test_pancohort, top N//2 diseases and bottom N//2 diseases seperately
    if args.is_pancohort and args.plot_box_logOR:
        os.makedirs(os.path.join(args.output_root, "plots", "logOR_pancohort"), exist_ok=True)

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_top, x='Disease', y='logOR_test_pancohort', hue='tag', hue_order=tag_order, palette="Set2")
        plt.title('logOR (pancohort) by Disease and Method', fontsize=16)
        plt.xlabel('Disease', fontsize=14)
        plt.ylabel('logOR', fontsize=14)
        plt.legend(title='Method', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        if args.save_plots:
            plt.savefig(f"{args.output_root}/plots/logOR_pancohort/top_{len(top_diseases)}_diseases.png", dpi=300)
        else:
            plt.show()

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_bottom, x='Disease', y='logOR_test_pancohort', hue='tag', hue_order=tag_order, palette="Set2")
        plt.title('logOR (pancohorts) by Disease and Method', fontsize=16)
        plt.xlabel('Disease', fontsize=14)
        plt.ylabel('logOR', fontsize=14)
        plt.legend(title='Method', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        if args.save_plots:
            plt.savefig(f"{args.output_root}/plots/logOR_pancohort/bottom_{len(bottom_diseases)}_diseases.png", dpi=300)
        else:
            plt.show()

    # %% Create prevalence vs probability plots for the disease cohort, for the best fold

    if args.plot_prevalence_prob_diseasecohort:
        os.makedirs(os.path.join(args.output_root, "plots", "prevalence_prob_diseasecohort"), exist_ok=True)

        #best fold
        for dis in sorted_diseases:
            r = f"{args.output_root}/{dis.replace(' ', '-')}_raw_results.pkl"

            with open(r, "rb") as pkl:
                res = pickle.load(pkl)

            best_AUC = 0
            best_fold = None
            for fold in res.keys():
                _, res_method = getIndividualProbs(res[fold]['GLM']['singlePRSCovar'], 'singlePRSCovar')
                if res_method['AUC_test'] > best_AUC:
                    best_AUC = res_method['AUC_test']
                    best_fold = fold

            probs = []
            tags = []
            for c in comparisons:
                pred_prob, _ = getIndividualProbs(res[best_fold][c[0]][c[1]], c[1])
                pred_prob.rename({'predicted': c[1]}, axis=1, inplace=True)
                probs.append(pred_prob)
                tags.append(c[2])
            probs = pd.concat(probs, axis=1)

            d = os.path.basename(r).replace('_raw_results.pkl', '')
            disease = pd.read_csv(f'{args.disease_root}/{d}.csv', low_memory=False, index_col="IID")
            disease["BinCAT_Disease"] = disease["BinCAT_Disease"].astype(int)

            combined = probs.join(disease[['BinCAT_Disease']])
            res_cols = [c for c in combined.columns if c!="BinCAT_Disease"]

            if args.save_plots:
                pth = f"{args.output_root}/plots/prevalence_prob_diseasecohort/bestfold_{d.replace(' ', '-')}.png"
            else:
                pth = ""
            plot_prevalence(combined, res_keys=res_cols, res_labels=tags, res_colours=colours[:len(tags)], res_fmts=markers[:len(tags)], col_disease='BinCAT_Disease', plot_title=f'Risk scores: {d.replace("-", " ")}', save_path=pth)

        # accross folds
        for dis in sorted_diseases:
            r = f"{args.output_root}/{dis.replace(' ', '-')}_raw_results.pkl"

            with open(r, "rb") as pkl:
                res = pickle.load(pkl)

            probs = []
            tags = []
            for c in comparisons:
                fold_probs = []
                for fold in res.keys():
                    pred_prob, _ = getIndividualProbs(res[fold][c[0]][c[1]], c[1])
                    fold_probs.append(pred_prob)
                fold_probs = pd.concat(fold_probs, axis=0)
                fold_probs.rename({'predicted': c[1]}, axis=1, inplace=True)
                probs.append(fold_probs)
                tags.append(c[2])
            probs = pd.concat(probs, axis=1)

            d = os.path.basename(r).replace('_raw_results.pkl', '')
            disease = pd.read_csv(f'{args.disease_root}/{d}.csv', low_memory=False, index_col="IID")
            disease["BinCAT_Disease"] = disease["BinCAT_Disease"].astype(int)

            combined = probs.join(disease[['BinCAT_Disease']])
            res_cols = [c for c in combined.columns if c!="BinCAT_Disease"]

            if args.save_plots:
                pth = f"{args.output_root}/plots/prevalence_prob_diseasecohort/acrossfolds_{d.replace(' ', '-')}.png"
            else:
                pth = ""
            plot_prevalence(combined, res_keys=res_cols, res_labels=tags, res_colours=colours[:len(tags)], res_fmts=markers[:len(tags)], col_disease='BinCAT_Disease', plot_title=f'Risk scores: {d.replace("-", " ")}', save_path=pth)

    # %% Create prevalence vs PRS scores plots for the pancohort, for the best single latent PRS

    if args.plot_prevalence_prs_pancohort:
        os.makedirs(os.path.join(args.output_root, "plots", "prevalence_prs_pancohort"), exist_ok=True)

        for dis in sorted_diseases:
            r = f"{args.output_root}/{dis.replace(' ', '-')}_raw_results.pkl"

            with open(r, "rb") as pkl:
                res = pickle.load(pkl)

            best_AUC = 0
            best_prs_latent = None
            for fold in res.keys():
                _, res_method = getIndividualProbs(res[fold]['GLM']['singlePRS'], 'singlePRS')
                if res_method['AUC_test'] > best_AUC:
                    best_AUC = res_method['AUC_test']
                    best_prs_latent = res_method.name.split("PRS:")[-1]
                
            pth_prs = f"{args.pth_prs_prefix}{best_prs_latent}{args.rds_pres_suffix}"

            data = pyreadr.read_r(pth_prs.replace(args.rds_tag_prs, args.tag_data))[None]
            prs = data[['IID']]
            r_obj = read_rds(pth_prs)
            prs["PRS"] = r_obj['data'][r_obj['attributes']['names']['data'].index(args.tag_prs)]['data']
            prs.set_index("IID", inplace=True)

            d = os.path.basename(r).replace('_raw_results.pkl', '')
            disease = pd.read_csv(f'{args.disease_root}/{d}.csv', low_memory=False, index_col="IID")
            combined = prs.join(disease[['BinCAT_Disease']])
            combined.fillna(0, inplace=True)
            combined["BinCAT_Disease"] = combined["BinCAT_Disease"].astype(int)

            if args.save_plots:
                pth = f"{args.output_root}/plots/prevalence_prs_pancohort/{d.replace(' ', '-')}.png"
            else:
                pth = ""

            if args.sex_stratified:
                data.IID = data.IID.astype(combined.index.dtype)
                IIDs_all = list(combined.index)
                IIDs_female = list(set(IIDs_all).intersection(set(list(data[data.Sex == "0"].IID))))
                IIDs_male = list(set(IIDs_all).intersection(set(list(data[data.Sex == "1"].IID))))
                combined.loc[IIDs_female, 'PRS_Female'] = combined.loc[IIDs_female, 'PRS']
                combined.loc[IIDs_male, 'PRS_Male'] = combined.loc[IIDs_male, 'PRS']
                pth = pth.replace(".png", "_sex_stratified.png")
                plot_prevalence(combined, res_keys=['PRS_Female', 'PRS', 'PRS_Male'], res_labels=['Female', 'All', 'Male'], res_colours=colours[:3], res_fmts=markers[:3], col_disease='BinCAT_Disease', plot_title=f'max(Single PRS) Risk scores: {d}', individual_prev_line=True, save_path=pth)
            else:
                plot_prevalence(combined, res_keys=['PRS'], res_labels=['max(Single PRS)'], res_colours=['blue'], res_fmts=['o'], col_disease='BinCAT_Disease', plot_title=f'Risk scores: {d}', save_path=pth)

    # %% Create prevalence vs probability plots for the pancohort, for the best fold
    
    if args.is_pancohort and args.plot_prevalence_prob_pancohort:
        os.makedirs(os.path.join(args.output_root, "plots", "prevalence_prob_pancohort"), exist_ok=True)

        for dis in sorted_diseases:
            r = f"{args.output_root}/{dis.replace(' ', '-')}_raw_results.pkl"

            with open(r, "rb") as pkl:
                res = pickle.load(pkl)

            best_AUC = 0
            best_fold = None
            for fold in res.keys():
                _, res_method = getIndividualProbs(res[fold]['GLM']['singlePRSCovar'], 'singlePRSCovar')
                if res_method['AUC_test'] > best_AUC:
                    best_AUC = res_method['AUC_test']
                    best_fold = fold

            probs = []
            tags = []
            for c in comparisons:
                pred_prob, _ = getIndividualProbs(res[best_fold][c[0]][c[1]], c[1], get_pancohort=True)
                pred_prob.rename({'predicted': c[1]}, axis=1, inplace=True)
                probs.append(pred_prob)
                tags.append(c[2])
            probs = pd.concat(probs, axis=1)

            d = os.path.basename(r).replace('_raw_results.pkl', '')
            disease = pd.read_csv(f'{args.disease_root}/{d}.csv', low_memory=False, index_col="IID")
            combined = probs.join(disease[['BinCAT_Disease']])
            combined.fillna(0, inplace=True)
            combined["BinCAT_Disease"] = combined["BinCAT_Disease"].astype(int)

            res_cols = [c for c in combined.columns if c!="BinCAT_Disease"]

            if args.save_plots:
                pth = f"{args.output_root}/plots/prevalence_prob_pancohort/bestfold_{d.replace(' ', '-')}.png"
            else:
                pth = ""
            plot_prevalence(combined, res_keys=res_cols, res_labels=tags, res_colours=colours[:len(tags)], res_fmts=markers[:len(tags)], col_disease='BinCAT_Disease', plot_title=f'Risk scores (Full cohort, excluding discovery and PRS training sets): {d.replace("-", " ")}', save_path=pth)