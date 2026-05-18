# %%
import sys
import os
import pickle
from collections import defaultdict
import argparse

from glob import glob
from tqdm import tqdm

import pandas as pd
import numpy as np

import pyreadr
from rds2py import read_rds

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

pd.options.mode.chained_assignment = None

# %%
def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

# %%
def fit_logistic_regression(data, trainIIDs, testIIDs, x_var='PRS_Score', col_y='BinCAT_Disease', cont_covars=None, cat_covars=None):
    data_train = data[data.index.isin(trainIIDs)]
    data_test = data[data.index.isin(testIIDs)]

    formula = f'{col_y} ~ {x_var}'
    if bool(x_var) and (bool(cont_covars) or bool(cat_covars)):
        formula += ' + '
    if cont_covars:
        formula += ' + '.join(cont_covars)
    if cat_covars:
        categorical_formula = ' + '.join([f'C({var})' for var in cat_covars])
        formula += ' + ' + categorical_formula
        
    model = smf.glm(formula, data=data_train, family=sm.families.Binomial())
    result = model.fit()
    data_train['predicted'] = result.predict(data_train)
    data_test['predicted'] = result.predict(data_test)
    auc_train = roc_auc_score(data_train[col_y], data_train['predicted'])
    logOR_train = logOR(data_train, col_y, data_train['predicted'])
    auc_test = roc_auc_score(data_test[col_y], data_test['predicted'])
    logOR_test = logOR(data_test, col_y, data_test['predicted'])
    
    return {
        'Coefficient': result.params[x_var] if bool(x_var) else None,
        'P-Value': result.pvalues[x_var] if bool(x_var) else None,
        'pred_probs_train': data_train['predicted'],
        'pred_probs_test': data_test['predicted'],
        'AUC_train': auc_train,
        'AUC_test': auc_test,
        'logOR_train': logOR_train,
        'logOR_test': logOR_test
    }

def fit_lasso_basic(data, trainIIDs, testIIDs, cols_X, col_y):
    data_train = data[data.index.isin(trainIIDs)]
    data_test = data[data.index.isin(testIIDs)]

    lasso_model = LogisticRegressionCV(cv=5, penalty='l1', solver='saga', max_iter=100, random_state=42)
    lasso_model.fit(data_train[cols_X], data_train[col_y])

    pred_probs_train = pd.DataFrame({"predicted": lasso_model.predict_proba(data_train[cols_X])[:, 1]}, index=data_train.index)
    pred_probs_test = pd.DataFrame({"predicted": lasso_model.predict_proba(data_test[cols_X])[:, 1]}, index=data_test.index)
    auc_train = roc_auc_score(data_train[col_y], pred_probs_train)
    logOR_train = logOR(data_train, col_y, pred_probs_train)
    auc_test = roc_auc_score(data_test[col_y], pred_probs_test)
    logOR_test = logOR(data_test, col_y, pred_probs_test)

    coef_results = pd.DataFrame({'Feature': cols_X, 'Coefficient': lasso_model.coef_[0]})
    
    return {
        'Coeff': coef_results,
        'pred_probs_train': pred_probs_train,
        'pred_probs_test': pred_probs_test,
        'AUC_train': auc_train,
        'AUC_test': auc_test,
        'logOR_train': logOR_train,
        'logOR_test': logOR_test
    }

def fit_lasso(data, trainIIDs, testIIDs, prs_cols, col_y='BinCAT_Disease', cont_covars=None, cat_covars=None, scale_prs=False, max_iter=1000, random_state=42, threads=5):
    data_train = data[data.index.isin(trainIIDs)]
    data_test = data[data.index.isin(testIIDs)]

    cols_X = prs_cols.copy()
    if scale_prs:
        scale_cols_cont = prs_cols.copy()
    else:
        scale_cols_cont = []

    if cont_covars:
        cols_X += cont_covars
        scale_cols_cont += cont_covars

    transformers = []
    if scale_cols_cont:
        transformers.append(('num', StandardScaler(), scale_cols_cont))

    if cat_covars:
        cols_X += cat_covars  
        transformers.append(('cat', OneHotEncoder(), cat_covars))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough' 
    )

    lasso_model = LogisticRegressionCV(cv=5, penalty='l1', solver='saga', max_iter=max_iter, random_state=random_state, n_jobs=threads)
    lasso_model = make_pipeline(preprocessor, lasso_model)

    lasso_model.fit(data_train[cols_X], data_train[col_y])

    pred_probs_train = pd.DataFrame({"predicted": lasso_model.predict_proba(data_train[cols_X])[:, 1]}, index=data_train.index)
    pred_probs_test = pd.DataFrame({"predicted": lasso_model.predict_proba(data_test[cols_X])[:, 1]}, index=data_test.index)
    auc_train = roc_auc_score(data_train[col_y], pred_probs_train)
    logOR_train = logOR(data_train, col_y, pred_probs_train)
    auc_test = roc_auc_score(data_test[col_y], pred_probs_test)
    logOR_test = logOR(data_test, col_y, pred_probs_test)

    features = lasso_model.named_steps['columntransformer'].get_feature_names_out()
    features = [f.split("__")[1] for f in features]
    coef_results = pd.DataFrame({'Feature': features, 'Coefficient': lasso_model.named_steps['logisticregressioncv'].coef_[0]})
    
    return {
        'Coeff': coef_results,
        'pred_probs_train': pred_probs_train,
        'pred_probs_test': pred_probs_test,
        'AUC_train': auc_train,
        'AUC_test': auc_test,
        'logOR_train': logOR_train,
        'logOR_test': logOR_test
    }

# %%
def logOR(data_test, col_y, pred_probs_test):
    try:
        data_test['predicted_prob'] = pred_probs_test
        data_test['quantile'] = pd.qcut(data_test['predicted_prob'], 5, labels=False, duplicates='drop') + 1
        odds_q3 = (data_test[data_test['quantile'] == 3][col_y].sum()) / \
                (data_test[data_test['quantile'] == 3][col_y].count() - data_test[data_test['quantile'] == 3][col_y].sum())
        odds_q5 = (data_test[data_test['quantile'] == 5][col_y].sum()) / \
                (data_test[data_test['quantile'] == 5][col_y].count() - data_test[data_test['quantile'] == 5][col_y].sum())
        return np.log(odds_q5 / odds_q3)
    except:
        return np.nan

def eval_glm(results):    
    resDF = pd.DataFrame(results.tolist(), index=results.index)
    resDF['Significant'] = resDF['P-Value'] < 0.05
    resDF.sort_values(by='AUC_test', ascending=False, inplace=True)
    return resDF

# %%
def getARGSParser():
    parser = argparse.ArgumentParser(description='MultiPRS Script')
    parser.add_argument('--prs_res_root', type=str, help='Path to PRS results root directory', default="/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc")
    parser.add_argument('--rds_pres_prefix', type=str, help='Prefix before the pheno name in the RDS file name', default="run_ext_basic_lw_gw_indep_FiltMAF_")
    parser.add_argument('--rds_pres_suffix', type=str, help='Suffix after the pheno name in the RDS file name', default=".fullDS.auto.mod.LDPred2.rds")
    parser.add_argument('--rds_tag_prs', type=str, help='tag PRS present in the rds file name', default="auto.mod")
    parser.add_argument('--tag_data', type=str, help='tag PRS model', default="resNdata.basic")
    parser.add_argument('--tag_prs', type=str, help='tag PRS inside the rds file', default="pred_auto")

    parser.add_argument('--disease_csv', type=str, help='Path to disease CSV file', default="/project/ukbblatent/clinicaldata/binary_disease_cohorts/F20208v3_nonMRI/caucasian/hypertension.csv")
    parser.add_argument('--col_disease', type=str, default='BinCAT_Disease')
    parser.add_argument('--min_sub', type=int, help='Minimum number of subjects must be present in the dataset. Set it to 0 or None to ignore this filter.', default=1000)

    parser.add_argument('--output_root', type=str, help='Path to store the output', default="/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/analyses/auto_lw_gw_20PC_1000it")

    parser.add_argument('--ext_covar', type=str, help='External covariates file is to be used, in addition to the fulldata file of the PRS. Keep it blank to only use the fulldata.', default="")
    parser.add_argument('--cont_covar_cols', type=str, help='Comma-separated list of continuous covariate columns', default='Age,BSA')
    parser.add_argument('--nPCs_covar', type=int, help='Number of principal components for covariates', default=20)
    parser.add_argument('--cat_covar_cols', type=str, help='Comma-separated list of categorical covariate columns', default='Sex')
    parser.add_argument('--lassoCV_max_iter', type=int, help='Maximum number of iterations for LassoCV', default=100)
    parser.add_argument('--threads', type=int, help='Number of threads', default=5)

    parser.add_argument('--do_singlePRS', action=argparse.BooleanOptionalAction, default=True, help='Run single PRS models')
    parser.add_argument('--do_singlePRSCovar', action=argparse.BooleanOptionalAction, default=True, help='Run single PRS + Covariate models')
    parser.add_argument('--do_covar', action=argparse.BooleanOptionalAction, default=True, help='Run Covariate models')
    parser.add_argument('--do_nonPCCovar', action=argparse.BooleanOptionalAction, default=True, help='Whether to additioanlly run covar-related PRS models without PCs as covariates (Only if nPCs_covar > 0)')

    parser.add_argument('--do_multiPRS', action=argparse.BooleanOptionalAction, default=True, help='Run multiPRS models')
    parser.add_argument('--do_multiPRSNorm', action=argparse.BooleanOptionalAction, default=True, help='Run multi normalised PRS models')
    parser.add_argument('--do_multiPRSCovar', action=argparse.BooleanOptionalAction, default=True, help='Run multiPRS + Covariate models')
    parser.add_argument('--do_multiPRSNormCovar', action=argparse.BooleanOptionalAction, default=True, help='Run multi normalised PRS + Covariate models')

    return parser

if __name__ == "__main__":
    parser = getARGSParser()
    args, _ = parser.parse_known_args()

    # %%
    os.makedirs(args.output_root, exist_ok=True)

    args.tag_disease = os.path.basename(args.disease_csv).replace(".csv", "")
    print(f"Working for disease: {args.tag_disease}....")

    if os.path.isfile(f"{args.output_root}/prs_auto.tsv") and os.path.isfile(f"{args.output_root}/fulldata.tsv"):
        print("The prs_auto.tsv and fulldata.tsv files already exist. Skipping the PRS data merging step. Loading the existing ones...")

        fulldata = pd.read_table(f"{args.output_root}/fulldata.tsv")
        prs = pd.read_table(f"{args.output_root}/prs_auto.tsv")

    else:
        files = glob(f"{args.prs_res_root}/{args.rds_pres_prefix}*{args.rds_pres_suffix}")
        print(f"Found {len(files)} PRS files")

        data = pyreadr.read_r(files[0].replace(args.rds_tag_prs, args.tag_data))[None]
        prs = data[['FID', 'IID']]

        for f in tqdm(files):
            r_obj = read_rds(f)
            prs_key = f'PRS:{os.path.basename(f).replace(args.rds_pres_prefix,"").replace(args.rds_pres_suffix, "")}'
            prs[prs_key] = r_obj['data'][r_obj['attributes']['names']['data'].index(args.tag_prs)]['data']
        prs = prs.copy() #to defragment

        fulldata = pd.merge(data, prs, on=['FID', 'IID'])

        prs.to_csv(f"{args.output_root}/prs_auto.tsv", sep="\t", index=False)
        fulldata.to_csv(f"{args.output_root}/fulldata.tsv", sep="\t", index=False)

    # %%
    disease = pd.read_csv(args.disease_csv, low_memory=False)    

    # %%
    #process the data and merge with disease

    if bool(args.ext_covar):
        ext_covars = pd.read_table(args.ext_covar)
        ext_covars = ext_covars.drop(columns=list(set(fulldata.columns).intersection(set(ext_covars.columns)) - {'IID'})) #drop the columns already exists inside the fulldata
        fulldata = pd.merge(fulldata, ext_covars, on='IID', how='inner')

    Xy = fulldata.merge(disease[['IID', args.col_disease]], on="IID")
    Xy.set_index("IID", inplace=True)
    IIDs = list(set(Xy.index))
    
    if bool(args.min_sub) and Xy.index.nunique() < args.min_sub*2:
        print(f"There are {Xy.index.nunique()} subjects present in the dataset, {args.min_sub}X2 is requested as minimum.")
        sys.exit(0)

    prs_Xy = Xy.filter(regex='^PRS|'+args.col_disease)
    prs_Xy = prs_Xy.reset_index().melt(id_vars=['IID', args.col_disease], var_name='PRS_Type', value_name='PRS_Score')
    prs_Xy.set_index('IID', inplace=True)

    prs_cols = [c for c in list(prs.columns) if c not in ['FID', 'IID']]

    # %% process the covariates
    
    nonPC_cont_covar_cols = args.cont_covar_cols.split(",")
    if bool(args.nPCs_covar):
        for i in range(1, args.nPCs_covar+1):
            args.cont_covar_cols += f',PC{i}'
    args.cont_covar_cols = args.cont_covar_cols.split(",")
    args.cat_covar_cols = args.cat_covar_cols.split(",")

    covars = fulldata[['IID']+args.cont_covar_cols+args.cat_covar_cols]
    covars.set_index('IID', inplace=True)
    prs_Xy_covar = prs_Xy.join(covars)

    # %%
    
    
    kf = KFold(n_splits=5, shuffle=True, random_state=1701)

    res_store = defaultdict(recursive_defaultdict)

    for fold, (train_index, test_index) in tqdm(enumerate(kf.split(IIDs), 1)):
        trainIIDs = [IIDs[index] for index in train_index]
        testIIDs = [IIDs[index] for index in test_index]
        print(f"Fold {fold}")

        res_store[f"Fold_{fold}"]['IDs']['train'] = trainIIDs
        res_store[f"Fold_{fold}"]['IDs']['test'] = testIIDs

        #Individual models

        #Predict the disease from individual PRS
        if args.do_singlePRS:
            results = prs_Xy.groupby('PRS_Type').apply(lambda x: fit_logistic_regression(x, trainIIDs, testIIDs, col_y=args.col_disease))
            res_store[f"Fold_{fold}"]['GLM']['singlePRS'] = eval_glm(results)

        #Predict the disase from individual PRS and covariates
        if args.do_singlePRSCovar:
            results = prs_Xy_covar.groupby('PRS_Type').apply(lambda x: fit_logistic_regression(x, trainIIDs, testIIDs, col_y=args.col_disease, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols))
            res_store[f"Fold_{fold}"]['GLM']['singlePRSCovar'] = eval_glm(results)

        #Predict the disease from the covariates
        if args.do_covar:
            res_store[f"Fold_{fold}"]['GLM']['covar'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, x_var='', col_y=args.col_disease, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols)

            #without PC covars
            if args.do_nonPCCovar and bool(args.nPCs_covar):
                res_store[f"Fold_{fold}"]['GLM']['nonPCCovar'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, x_var='', col_y=args.col_disease, cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols)

        ################################
        #Lasso models

        #multiPRS
        if args.do_multiPRS:
            res_store[f"Fold_{fold}"]['Lasso']['multiPRS'] = fit_lasso(Xy, trainIIDs, testIIDs, prs_cols=prs_cols, col_y=args.col_disease, scale_prs=False, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads)

        #multiPRS normalised
        if args.do_multiPRSNorm:
            res_store[f"Fold_{fold}"]['Lasso']['multiPRSNorm'] = fit_lasso(Xy, trainIIDs, testIIDs, prs_cols=prs_cols, col_y=args.col_disease, scale_prs=True, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads)

        #multiPRS + covar
        if args.do_multiPRSCovar:
            res_store[f"Fold_{fold}"]['Lasso']['multiPRSCovar'] = fit_lasso(Xy, trainIIDs, testIIDs, prs_cols=prs_cols, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, scale_prs=False, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads)
            
            #without PC covars
            if args.do_nonPCCovar and bool(args.nPCs_covar):
                res_store[f"Fold_{fold}"]['Lasso']['multiPRSnonPCCovar'] = fit_lasso(Xy, trainIIDs, testIIDs, prs_cols=prs_cols, cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, scale_prs=False, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads)

        #multiPRS + covar
        if args.do_multiPRSNormCovar:
            res_store[f"Fold_{fold}"]['Lasso']['multiPRSNormCovar'] = fit_lasso(Xy, trainIIDs, testIIDs, prs_cols=prs_cols, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, scale_prs=True, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads)
            
            #without PC covars
            if args.do_nonPCCovar and bool(args.nPCs_covar):
                res_store[f"Fold_{fold}"]['Lasso']['multiPRSNormnonPCCovar'] = fit_lasso(Xy, trainIIDs, testIIDs, prs_cols=prs_cols, cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, scale_prs=True, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads)

        #covar
        if args.do_covar:
            res_store[f"Fold_{fold}"]['Lasso']['covar'] = fit_lasso(Xy, trainIIDs, testIIDs, prs_cols=[], cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, scale_prs=False, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads)
            
            #without PC covars
            if args.do_nonPCCovar and bool(args.nPCs_covar):
                res_store[f"Fold_{fold}"]['Lasso']['nonPCCovar'] = fit_lasso(Xy, trainIIDs, testIIDs, prs_cols=[], cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, scale_prs=False, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads)

    with open(f"{args.output_root}/{args.tag_disease}_raw_results.pkl", "wb") as f:
        pickle.dump(res_store, f)

    # %%
    score_store = []

    for fold in res_store.keys():

        for method in res_store[fold].keys():
            if method == 'IDs':
                continue
            for res_type in res_store[fold][method].keys():
                if type(res_store[fold][method][res_type]) is dict:
                    datum = {
                        "fold": fold.replace("Fold_", ""),
                        "method": method,
                        "res_type": res_type,
                        "AUC_test": res_store[fold][method][res_type]['AUC_test'],
                        "logOR_test": res_store[fold][method][res_type]['logOR_test']
                    }
                else:
                    datum = {
                        "fold": fold.replace("Fold_", ""),
                        "method": method,
                        "res_type": res_type,
                        "AUC_test": res_store[fold][method][res_type]['AUC_test'].max(),
                        "logOR_test": res_store[fold][method][res_type]['logOR_test'].max()
                    }
                score_store.append(datum)

    score_store = pd.DataFrame(score_store)
    score_store.to_csv(f"{args.output_root}/{args.tag_disease}_models_test_scores.tsv", sep="\t", index=False)    
    print(f"{args.output_root}/{args.tag_disease}_models_test_scores.tsv")

