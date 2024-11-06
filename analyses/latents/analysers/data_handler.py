import sys
# import argparse
import os
# import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from functools import partial
# import statsmodels.api as sm
# from sklearn.feature_selection import RFE, SelectFromModel
# from sklearn.compose import ColumnTransformer
# from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, LassoCV, Lasso
# from sklearn.metrics import mean_squared_error, r2_score, classification_report
# from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
# from scipy import stats
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# import warnings
# from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

sys.path.insert(0, os.getcwd()) #to handle the sub-foldered structure of the tricoder

from utils.python_utils import DotDict, reduce_numeric_precision
from H5tools.traverse_embH5 import process_embs
from UKBB.ukbb_postprocess_errors import correct_errors_raw
# from analyses.latents.utils import remove_high_vif_features

def transform_complex_columns(df, columns, mode):
    if mode == 'real':
        df[columns] = df[columns].applymap(lambda x: x.real)
    elif mode == 'imag':
        df[columns] = df[columns].applymap(lambda x: x.imag)
    elif mode == 'mag':
        df[columns] = df[columns].applymap(abs)
    elif mode == 'phase':
        df[columns] = df[columns].applymap(np.angle)
    elif mode == 'cartesian':
        df[[f'{col}_real' for col in columns]] = df[columns].applymap(lambda x: x.real)
        df[[f'{col}_imag' for col in columns]] = df[columns].applymap(lambda x: x.imag)
        df.drop(columns=columns, inplace=True)
    elif mode == 'polar':
        df[[f'{col}_mag' for col in columns]] = df[columns].applymap(abs)
        df[[f'{col}_phase' for col in columns]] = df[columns].applymap(np.angle)
        df.drop(columns=columns, inplace=True)
    elif mode == 'dualcoords':
        df[[f'{col}_real' for col in columns]] = df[columns].applymap(lambda x: x.real)
        df[[f'{col}_imag' for col in columns]] = df[columns].applymap(lambda x: x.imag)
        df[[f'{col}_mag' for col in columns]] = df[columns].applymap(abs)
        df[[f'{col}_phase' for col in columns]] = df[columns].applymap(np.angle)
        df.drop(columns=columns, inplace=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return df

class AnalysersDataHandler:
    def __init__(self, args):
        self.__dict__.update(args)
        self.res_collect = defaultdict(partial(defaultdict, dict))  
        
        self.df_test = None  # variable will be set only if held-out test is provided
        if bool(self.embH5):
            self.df = process_embs(args=DotDict({"in_path": self.embH5, "prep_Zs": True}), save_npy=False).set_index("subID")
            self.df['instanceID'] = self.df.instanceID.str.split("_").str[0].astype('int8')
            self.latent_factors = [c for c in self.df.columns if c.startswith("Z")]
            if bool(self.embH5_heldout):  # held-out test subjects (if present)
                self.df_test = process_embs(args=DotDict({"in_path": self.embH5_heldout, "prep_Zs": True}), save_npy=False).set_index("subID")
                self.df_test['instanceID'] = self.df_test.instanceID.str.split("_").str[0].astype('int8')
        elif bool(self.res_path):
            self.df = pd.read_table(self.res_path)
            self.latent_factors = [f'Z{i}' for i in range(len(self.df.columns)-2)]
            self.df.columns = ['FID', 'IID'] + self.latent_factors
            self.df.set_index('IID', inplace=True) 
            if bool(self.res_heldout_path):  # held-out test subjects (if present)
                self.df_test = pd.read_table(self.res_heldout_path)
                self.df_test.set_index('IID', inplace=True)
        elif bool(self.merged_path):
            self.df = pd.read_table(self.merged_path).set_index('IID')
            self.latent_factors = [c for c in self.df.columns]
            if bool(self.merged_heldout_path):  # held-out test subjects (if present)
                self.df_test = pd.read_table(self.merged_heldout_path).set_index('IID')
        else:  # then we use other predictors (not latent factors)
            self.df = pd.read_table(self.pred_path).set_index('f.eid')
            self.latent_factors = [c for c in self.df.columns]
            if bool(self.pred_heldout_path):  # held-out test subjects (if present)
                self.df_test = pd.read_table(self.pred_heldout_path).set_index('f.eid')

        self.df.index = self.df.index.astype('int32')
        if self.df_test is not None:
            self.df_test.index = self.df_test.index.astype('int32')

        self.latent_factors = [lf for lf in self.latent_factors if lf not in self.predictors2ignore]  # exclude predictors to ignore
        self.df = self.df[self.latent_factors+["instanceID"] if bool(self.embH5) and self.instance_merge else self.latent_factors]
        if self.df_test is not None:
            self.df_test = self.df_test[self.latent_factors+["instanceID"] if bool(self.embH5) and self.instance_merge else self.latent_factors]
        
        if "force_complex" in args and bool(args["force_complex"]):
            self.df[self.latent_factors] = self.df[self.latent_factors] + 0j
            if self.df_test is not None:
                self.df_test[self.latent_factors] = self.df_test[self.latent_factors] + 0j

        if np.issubdtype(self.df[self.latent_factors[0]], np.complexfloating):
            #latent factors are complex-valued
            self.is_complex = True
            if "complex_mode" in args:
                self.df = transform_complex_columns(self.df.copy(), self.latent_factors, self.complex_mode)
                self.df_test = transform_complex_columns(self.df_test.copy(), self.latent_factors, self.complex_mode) if self.df_test is not None else None
                if self.complex_mode in ['cartesian', 'polar', 'dualcoords']:
                    self.latent_factors = [c for c in self.df.columns if c.endswith("_real") or c.endswith("_imag") or c.endswith("_mag") or c.endswith("_phase")]
        else:
            self.is_complex = False

        # Covariates
        if bool(self.cov_path) and bool(self.cov_cols):
            attribs = self.cov_cols.split(',')
            self.cov_bincat = self.cov_bincat.split(',')
            self.cov_cat = self.cov_cat.split(',')
            df_temp = pd.read_table(self.cov_path)[["IID", "MRI_Visit"]+attribs].rename(columns=(lambda x: 'BinCAT_' + x if x in self.cov_bincat else ('CAT_' + x if x in self.cov_cat else x))).rename(columns=(lambda x: x + '_COV'))  # read file of covariates
            self.df = self.df.reset_index().rename({"index":"subID"}, axis=1).merge(df_temp, left_on=['subID', 'instanceID'], right_on=['IID_COV', 'MRI_Visit_COV'], how='inner').drop(columns=['IID_COV', 'MRI_Visit_COV']).set_index('subID')
            self.df_test = self.df_test.reset_index().rename({"index":"subID"}, axis=1).merge(df_temp, left_on=['subID', 'instanceID'], right_on=['IID_COV', 'MRI_Visit_COV'], how='inner').drop(columns=['IID_COV', 'MRI_Visit_COV']).set_index('subID') if self.df_test is not None else None            
            self.cov_cols = []
            for att in attribs:
                if att in self.cov_bincat:
                    self.cov_cols.append(f"BinCAT_{att}_COV")
                elif att in self.cov_cat:
                    self.cov_cols.append(f"CAT_{att}_COV")
                else:
                    self.cov_cols.append(f"{att}_COV")

        self.df = reduce_numeric_precision(self.df)
        self.df_test = reduce_numeric_precision(self.df_test) if self.df_test is not None else None

        ## Additional features for prediction
        self.df_add = None
        if bool(self.add_feat_path):
            self.df_add = pd.read_table(self.add_feat_path).set_index('f.eid')
            self.df_add.index = self.df_add.index.astype('int32')
            self.df_add = self.df_add.drop(columns=self.add_feat2ignore, errors='ignore')  # drop columns to ignore
            self.df_add = self.df_add.sort_index()

        self.tsv_name = self.model_tag
        self.tsv_files = self.tsv_files.split(',')
        for tsv_file in self.tsv_files:
            tsv_file = tsv_file if tsv_file.endswith('.tsv') else f"{tsv_file}.tsv"
            df_temp = pd.read_table(f"{self.path_tsvs}/{tsv_file}" if bool(self.path_tsvs) else tsv_file).set_index('f.eid')
            df_temp.index = df_temp.index.astype('int32')
            df_temp = reduce_numeric_precision(df_temp)

            if self.instance_merge:
                df_temp = correct_errors_raw(df_temp)
                df_temp = df_temp.reset_index().melt(id_vars='f.eid', var_name='measurement', value_name='value')
                df_temp['instanceID'] = df_temp['measurement'].str.extract(r'\.(\d+\.\d+)$')[0].str.replace(".0","").astype("int8") #let's assume there is no subtype of the instance
                df_temp['measurement'] = df_temp['measurement'].str.replace(r'\.\d+\.\d+$', '', regex=True)
                df_temp = df_temp.dropna().pivot_table(index=['f.eid', 'instanceID'], 
                                    columns='measurement', 
                                    values='value').reset_index().set_index('f.eid')
                df_temp.columns.name = None
                instances_df = []
                instances_df_test = []
                for inst in self.df.instanceID.unique():
                    instances_df.append(self.df[self.df.instanceID==inst].join(df_temp[df_temp.instanceID==inst], how="inner", rsuffix="_tmp"))
                    if self.df_test is not None:
                        instances_df_test.append(self.df_test[self.df_test.instanceID==inst].join(df_temp[df_temp.instanceID==inst], how="inner"))
                self.df = pd.concat(instances_df)
                del self.df["instanceID_tmp"]
                if self.df_test is not None:
                    self.df_test = pd.concat(instances_df_test) 
                    del self.df_test["instanceID_tmp"]
            else:
                self.df = self.df.join(df_temp, how="inner")
                self.df_test = self.df_test if self.df_test is None else self.df_test.join(df_temp, how="inner")
            self.tsv_name = f"{self.tsv_name}_{os.path.basename(tsv_file).split('.')[0]}"

        if bool(self.embH5) and self.instance_merge:
            self.df = self.df.reset_index().rename({"index":"subID"}, axis=1).set_index(["subID", "instanceID"])
            self.df_test = self.df_test.reset_index().set_index(["index", "instanceID"]) if self.df_test is not None else None

        self.attributes2ignore = [ai for ai in self.attributes2ignore if ai in self.df.columns] + [ai for ai in self.df.columns if ai.endswith("_COV")]
        self.binary_attributes = [c for c in self.df.columns if (c.startswith("BinCAT") and c not in self.attributes2ignore+self.latent_factors)]
        self.multi_class_attributes = [c for c in self.df.columns if (c.startswith("CAT") and c not in self.attributes2ignore+self.latent_factors)]
        self.continuous_attributes = [c for c in self.df.columns if c not in self.attributes2ignore+self.latent_factors+self.multi_class_attributes+self.binary_attributes]
        if bool(self.path_norm) and bool(self.attribute_norm):  # normalising attribute
            normal_attributes = []
            df_temp = pd.read_table(self.path_norm).set_index('f.eid')[self.attribute_norm]
            self.df = self.df.join(df_temp, how="inner", rsuffix="_")
            self.df_test = self.df_test if self.df_test is None else self.df_test.join(df_temp, how="inner", rsuffix="_")
            for c in self.continuous_attributes:
                c_norm = c + "_norm"
                self.df[c_norm] = self.df[c] / self.df[self.attribute_norm]  # normalise
                if self.df_test is not None:
                    self.df_test[c_norm] = self.df_test[c] / self.df_test[self.attribute_norm]  # normalise
                normal_attributes.append(c_norm)
            self.continuous_attributes.extend(normal_attributes)
            self.tsv_name = f"Norm{self.attribute_norm}_{self.tsv_name}"
        self.attributes = self.continuous_attributes + self.binary_attributes + self.multi_class_attributes
        self.categorical_attributes = self.binary_attributes + self.multi_class_attributes

        #self.df = self.df.sample(frac=1, random_state=1701)  # shuffle dataset rows
        self.df = self.df.sort_index()