import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats

from utils import categorise_corrcoef_strength, PlotNSaveHeatmap
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

class CorrAnalyses:
    def __init__(self, args):
        self.__dict__.update(args)
        
        if bool(self.res_path):
            self.df = pd.DataFrame(np.load(f"{self.res_path}/emb.npy", allow_pickle=True))
            self.latent_factors = [f'Z{i}' for i in range(len(self.df.columns)-4)]
            self.df.columns = ['subID','fieldID','instanceID','data_tag'] + self.latent_factors
            self.df.set_index(['subID'], inplace=True)
            self.df.index = self.df.index.astype(int)
            self.df.drop(columns=['fieldID','instanceID','data_tag'], inplace=True) #For now, this is fine. If we want to do multi-modal, we need to re-consider this
            self.df = self.df.astype(float)
        else: #then merged_path is not empty
            self.df = pd.read_table(f"{self.merged_path}/merged_latents_raw.tsv").set_index('FID')
            self.df.drop(columns=['IID'], inplace=True)
            self.latent_factors = [c for c in self.df.columns if c.startswith("Z")]

        self.tsv_name = self.model_tag
        self.tsv_files = self.tsv_files.split(',')
        for tsv_file in self.tsv_files:
            tsv_file = tsv_file if tsv_file.endswith('.tsv') else f"{tsv_file}.tsv"
            df_temp = pd.read_table(f"{self.path_tsvs}/{tsv_file}" if bool(self.path_tsvs) else tsv_file).set_index('f.eid')
            self.df = self.df.join(df_temp, how="inner")
            self.tsv_name = f"{self.tsv_name}_{os.path.basename(tsv_file).split('.')[0]}"

        self.attributes2ignore = [ai for ai in self.attributes2ignore if ai in self.df.columns]
        self.binary_attributes = [c for c in self.df.columns if c.startswith("BinCAT")]
        self.multi_class_attributes = [c for c in self.df.columns if c.startswith("CAT")]
        self.continuous_attributes = [c for c in self.df.columns if c not in self.attributes2ignore+self.latent_factors+self.multi_class_attributes+self.binary_attributes]
        
    def calculate_correlations(self):
        correlations = {}

        # Continuous attributes
        for factor in self.latent_factors:
            for var in self.continuous_attributes:
                pearson_corr, pearson_pval = stats.pearsonr(self.df[factor], self.df[var])
                spearman_corr, spearman_pval = stats.spearmanr(self.df[factor], self.df[var])
                correlations[(factor, var, 'pearson')] = (pearson_corr, pearson_pval)
                correlations[(factor, var, 'spearman')] = (spearman_corr, spearman_pval)

        # Binary categorical attributes
        for factor in self.latent_factors:
            for var in self.binary_attributes:
                point_biserial_corr, point_biserial_pval = stats.pointbiserialr(self.df[factor], self.df[var])
                correlations[(factor, var, 'point_biserial')] = (point_biserial_corr, point_biserial_pval)

        # Multi-class categorical atttributes
        for factor in self.latent_factors:
            for var in self.multi_class_attributes:
                model = ols(f"{factor} ~ C({var})", data=self.df).fit()
                anova_table = anova_lm(model)
                eta_squared = anova_table['sum_sq'][0] / (anova_table['sum_sq'][0] + anova_table['sum_sq'][1])
                eta_squared_pval = anova_table['PR(>F)'][0]
                correlations[(factor, var, 'eta_squared')] = (eta_squared, eta_squared_pval)

        rows = []
        for key, value in correlations.items():
            corr_coeff, p_val = value
            rows.append([key[0], key[1], key[2], corr_coeff, p_val])

        self.correlations = pd.DataFrame(rows, columns=['Factor', 'Attribute', 'Method', 'Coefficient', 'p-value'])

    def categorise_correlations(self, alpha=0.05):        
        self.correlations['isSignificant'] = self.correlations['p-value'] < alpha
        self.correlations['Strength'] = self.correlations['Coefficient'].apply(categorise_corrcoef_strength)

    def obtain_total_correlations(self, gen_heatmaps=True):
        def stack_corr_matrix(corr_matrix):
            stacked_df = pd.DataFrame(np.where(np.eye(corr_matrix.shape[0]), np.nan, corr_matrix.values),
                                      index=corr_matrix.index, columns=corr_matrix.columns).stack().reset_index()
            stacked_df.columns = ['Attrib1', 'Attrib2', 'Coefficient']
            stacked_df['Strength'] = stacked_df['Coefficient'].apply(categorise_corrcoef_strength)
            stacked_df[["Attrib1", "Attrib2"]] = pd.DataFrame(np.sort(stacked_df[["Attrib1", "Attrib2"]].values, axis=1))
            return stacked_df.drop_duplicates()
        
        # Convert the categorial variables to one-hot encoding
        df_1HOT = pd.get_dummies(self.df, columns=[col for col in self.df.columns if 'CAT' in col])        
        df_1HOT = df_1HOT.drop(columns=self.attributes2ignore)

        pearson_matrix = df_1HOT.corr()
        spearman_corr = df_1HOT.corr(method='spearman')
        kendall_corr = df_1HOT.corr(method='kendall')

        stacked_pearson_matrix = stack_corr_matrix(pearson_matrix)
        stacked_spearman_corr = stack_corr_matrix(spearman_corr)
        stacked_kendall_corr = stack_corr_matrix(kendall_corr)

        #combine those three dataframes into one, using the name of the correlation method as a column
        stacked_corrs = pd.concat([stacked_pearson_matrix.assign(Method='pearson'),
                                         stacked_spearman_corr.assign(Method='spearman'),
                                         stacked_kendall_corr.assign(Method='kendall')])
        stacked_corrs.to_csv(os.path.join(self.out_path, f"{self.tsv_name}_total_correlations.tsv"), sep='\t')

        if gen_heatmaps:
            PlotNSaveHeatmap(pearson_matrix, 'Correlation Matrix Heatmap', fig_size=2500, font_size=8, rootpath=self.out_path, filename=f"{self.tsv_name}_total_pearson")
            PlotNSaveHeatmap(spearman_corr, 'Spearman Rank Correlation Matrix Heatmap', fig_size=2500, font_size=8, rootpath=self.out_path, filename=f"{self.tsv_name}_total_spearmanRank")
            PlotNSaveHeatmap(kendall_corr, 'Kendall Tau Rank Correlation Matrix Heatmap', fig_size=2500, font_size=8, rootpath=self.out_path, filename=f"{self.tsv_name}_total_kendalltauRank")
    
    def save_results(self):
        if bool(self.out_path):
            os.makedirs(self.out_path, exist_ok=True)
            self.correlations.to_csv(os.path.join(self.out_path, f"{self.tsv_name}_correlations.tsv"), sep='\t')

def process_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--res_path', action="store", default="", help="Folder containining the results of the model, emb.h5.")
    parser.add_argument('--merged_path', action="store", default="", help="Folder containining the results of a merge, merged_latents_raw.tsv.")
    
    parser.add_argument('--model_tag', action="store", default="prova", help="Tag to identify the model.")

    parser.add_argument('--out_path', action="store", default="", help="Path to store the results. Make it blank if if it's desired to not store the results, rather just print them on the console.")
    
    parser.add_argument('--path_tsvs', action="store", default="", help="Location of the tsv files, to be used for analyses. Leave it blank if fully-qualified paths are supplied in --tsv_files.")
    parser.add_argument('--tsv_files', action="store", default="", help="Coma-separated list of tsv files - to be joined and considered. To consider in one analysis, seperate by coma, to be considered in seperate ones, seperate by semicolon.")
    
    parser.add_argument('--attributes2ignore', action="store", default="", help="Coma-separated list of attributes to ignore.") #Reasons for the default vals: Birth_Month makes no sense, Smoking_Imaging is considered, Ethnicity is mostly white, BMI_Imaging is considered, MRIvisit shouldn't have any impact (but we should check it out in the future)
    
    parser.add_argument('--use_feature_scaling', action=argparse.BooleanOptionalAction, default=True, help="Whether to use feature scaling or not.")

    parser.add_argument('--run_focused_corr', action=argparse.BooleanOptionalAction, default=True, help="Whether to run the focused correlations (i.e. Zs vs attributes).")
    parser.add_argument('--run_total_corr', action=argparse.BooleanOptionalAction, default=False, help="Whether to run the total correlations (i.e. all vs all).")
    parser.add_argument('--gen_heatmaps_total_corr', action=argparse.BooleanOptionalAction, default=False, help="Whether to generate the heatmaps from the total correlations.")

    args, unknown_args = parser.parse_known_args()

    args.attributes2ignore = args.attributes2ignore.split(',')

    return args, unknown_args

def main():
    args, unknown_args = process_arguments()
    
    if bool(args.res_path) and bool(args.merged_path):
        raise ValueError("Both res_path and merged_path cannot be specified at the same time.")
    
    tsv_file_combos = args.tsv_files.split(';')
    for tsv_file_combo in tsv_file_combos:
        args.tsv_files = tsv_file_combo
        analyser = CorrAnalyses(vars(args))

        if args.run_focused_corr:
            analyser.calculate_correlations()
            analyser.categorise_correlations()
            analyser.save_results()

        if args.run_total_corr:
            analyser.obtain_total_correlations(gen_heatmaps=args.gen_heatmaps_total_corr)

if __name__ == "__main__":
    main()