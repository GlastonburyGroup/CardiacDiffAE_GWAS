# %%
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GridSearchCV

from group_lasso import GroupLasso, LogisticGroupLasso

from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector

from skimage.filters import threshold_otsu

pd.options.mode.chained_assignment = None

# %%
def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

def binary_accuracy(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    return accuracy_score(y_true, y_pred_binary)

# %% Custom models
class ToLassoORnotToLasso(BaseEstimator, ClassifierMixin):
    def __init__(self, idx_lasso_features, idx_non_lasso_features, lasso_params=None, non_lasso_nonCV=True, logistic_params=None):
        self.idx_lasso_features = idx_lasso_features
        self.idx_non_lasso_features = idx_non_lasso_features
        self.lasso_params = lasso_params if lasso_params is not None else {}
        self.logistic_params = logistic_params if logistic_params is not None else {}
        self.lasso_model = LogisticRegressionCV(penalty='l1', solver='saga', **self.lasso_params)
        if non_lasso_nonCV:
            self.non_lasso_model = LogisticRegression(**self.logistic_params)
        else:
            self.non_lasso_model = LogisticRegressionCV(penalty='l2', solver='lbfgs', **self.lasso_params)

    def fit(self, X, y):
        X_lasso = X[:, self.idx_lasso_features]
        X_non_lasso = X[:, self.idx_non_lasso_features]
        
        self.lasso_model.fit(X_lasso, y)
        self.non_lasso_model.fit(X_non_lasso, y)

        self._coef_ = np.zeros((1, X.shape[1]))
        self._coef_[0, self.idx_lasso_features] = self.lasso_model.coef_
        self._coef_[0, self.idx_non_lasso_features] = self.non_lasso_model.coef_

        self._intercept_ = self.lasso_model.intercept_ + self.non_lasso_model.intercept_

        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        # Compute logits for both sets of features
        logit_lasso = np.dot(X[:, self.idx_lasso_features], self.coef_[0, self.idx_lasso_features])
        logit_no_lasso = np.dot(X[:, self.idx_non_lasso_features], self.coef_[0, self.idx_non_lasso_features])
        
        # Combine logits
        combined_logit = logit_lasso + logit_no_lasso + self.intercept_
        
        # Apply sigmoid function
        probabilities = 1 / (1 + np.exp(-combined_logit))
        return np.vstack([1 - probabilities, probabilities]).T

    @property
    def coef_(self):
        if self._coef_ is None:
            raise AttributeError("Coefficient has not been set. Fit the model first.")
        return self._coef_

    @coef_.setter
    def coef_(self, value):
        self._coef_ = value

    @property
    def intercept_(self):
        if self._intercept_ is None:
            raise AttributeError("Intercept has not been set. Fit the model first.")
        return self._intercept_

    @intercept_.setter
    def intercept_(self, value):
        self._intercept_ = value

class LassoSteps(BaseEstimator, ClassifierMixin):
    def __init__(self, idx_lasso_features, idx_non_lasso_features, lasso_params=None, logistic_params=None, mode=0):
        self.idx_lasso_features = idx_lasso_features
        self.idx_non_lasso_features = idx_non_lasso_features
        self.lasso_params = lasso_params if lasso_params is not None else {}
        self.logistic_params = logistic_params if logistic_params is not None else {}
        self.mode = mode
        if self.mode in [0,1]:
            pipe_lasso = make_pipeline(ColumnSelector(cols=self.idx_lasso_features),
                                        LogisticRegressionCV(penalty='l1', solver='saga', **self.lasso_params))
            pipe_nonlasso = make_pipeline(ColumnSelector(cols=self.idx_non_lasso_features),
                                            LogisticRegression(**self.logistic_params))
            self.model = StackingClassifier(classifiers=[pipe_lasso, pipe_nonlasso],
                                            meta_classifier=LogisticRegression(**self.logistic_params), 
                                            use_probas=True if mode==1 else 0)
        else:
            self.lasso_model = LogisticRegressionCV(penalty='l1', solver='saga', **self.lasso_params)
            self.model = LogisticRegression(**self.logistic_params)

    def fit(self, X, y):
        if self.mode in [0,1]:
            self.model.fit(X, y)
        else:
            X_lasso = X[:, self.idx_lasso_features]
            self.lasso_model.fit(X_lasso, y)
            if self.mode == 2:
                lasso_out = self.lasso_model.predict_proba(X_lasso)[:, 1]
            else:
                lasso_out = np.dot(X[:, self.idx_lasso_features], self.lasso_model.coef_[0])
            X_non_lasso = X[:, self.idx_non_lasso_features]
            X_combined = np.hstack([lasso_out.reshape(-1, 1), X_non_lasso])
            self.model.fit(X_combined, y)
        return self

    def predict(self, X):
        if self.mode in [0,1]:
            return self.model.predict(X)
        else:
            X_lasso = X[:, self.idx_lasso_features]
            if self.mode == 2:
                lasso_out = self.lasso_model.predict_proba(X_lasso)[:, 1]
            else:
                lasso_out = np.dot(X_lasso, self.lasso_model.coef_[0])
            X_non_lasso = X[:, self.idx_non_lasso_features]
            X_combined = np.hstack([lasso_out.reshape(-1, 1), X_non_lasso])
            return self.model.predict(X_combined)

    def predict_proba(self, X):
        if self.mode in [0,1]:
            return self.model.predict_proba(X)
        else:
            X_lasso = X[:, self.idx_lasso_features]
            if self.mode == 2:
                lasso_out = self.lasso_model.predict_proba(X_lasso)[:, 1]
            else:
                lasso_out = np.dot(X_lasso, self.lasso_model.coef_[0])
            X_non_lasso = X[:, self.idx_non_lasso_features]
            X_combined = np.hstack([lasso_out.reshape(-1, 1), X_non_lasso])
            return self.model.predict_proba(X_combined)

# %%
def fit_logistic_regression(data, trainIIDs, testIIDs, IIDs_nodisease=None, x_var='PRS_Score', prs_cols=[], col_y='BinCAT_Disease', cont_covars=None, cat_covars=None, use_scaler=False):
    data_train = data[data.index.isin(trainIIDs)]
    data_test = data[data.index.isin(testIIDs)]

    formula = f'{col_y} ~ {x_var}'
    if bool(x_var) and (bool(cont_covars) or bool(cat_covars)):
        formula += ' + '

    if prs_cols:
        prs_cols_formula = [f'Q("{col}")' for col in prs_cols]
        formula += ' + '.join(prs_cols_formula)
        if bool(cont_covars) or bool(cat_covars):
            formula += ' + '

    if cont_covars:
        cont_covars_formula = [f'Q("{col}")' for col in cont_covars]
        formula += ' + '.join(cont_covars_formula)

    if cat_covars:
        cat_covars_formula = ' + '.join([f'C({var})' for var in cat_covars])
        formula += ' + ' + cat_covars_formula

    if use_scaler:
        scaler = StandardScaler()
        cols2norm = [x_var] if bool(x_var) else []
        if prs_cols:
            cols2norm += prs_cols
        if cont_covars:
            cols2norm += cont_covars
        data_train[cols2norm] = scaler.fit_transform(data_train[cols2norm])
        data_test[cols2norm] = scaler.transform(data_test[cols2norm])
        
    model = smf.glm(formula, data=data_train, family=sm.families.Binomial())
    result = model.fit()
    data_train['predicted'] = result.predict(data_train)
    data_test['predicted'] = result.predict(data_test)
    auc_train = roc_auc_score(data_train[col_y], data_train['predicted'])
    logOR_train = logOR(data_train, col_y, data_train['predicted'])
    auc_test = roc_auc_score(data_test[col_y], data_test['predicted'])
    logOR_test = logOR(data_test, col_y, data_test['predicted'])

    f1_test = getF1(data_test[col_y], data_test['predicted'], method="at50")
    
    res = {
        'Coefficients': result.params,
        'P-Values': result.pvalues,
        'pred_probs_train': data_train['predicted'],
        'pred_probs_test': data_test['predicted'],
        'AUC_train': auc_train,
        'AUC_test': auc_test,
        'logOR_train': logOR_train,
        'logOR_test': logOR_test,
        'F1_test': f1_test
    }

    if bool(IIDs_nodisease):
        data_test_pancohort = data[data.index.isin(testIIDs+IIDs_nodisease)]
        if use_scaler:
            data_test_pancohort[cols2norm] = scaler.transform(data_test_pancohort[cols2norm])
        data_test_pancohort['predicted'] = result.predict(data_test_pancohort)
        logOR_test_pancohort = logOR(data_test_pancohort, col_y, data_test_pancohort['predicted'])
        res['pred_probs_test_pancohort'] = data_test_pancohort['predicted']
        res['logOR_test_pancohort'] = logOR_test_pancohort

    return res

def fit_lasso_basic(data, trainIIDs, testIIDs, cols_X, col_y, IIDs_nodisease=None):
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

    f1_test = f1_score(data_test[col_y], lasso_model.predict(data_test[cols_X]))

    coef_results = pd.DataFrame({'Feature': cols_X, 'Coefficient': lasso_model.coef_[0]})
    
    res = {
        'Coeff': coef_results,
        'pred_probs_train': pred_probs_train,
        'pred_probs_test': pred_probs_test,
        'AUC_train': auc_train,
        'AUC_test': auc_test,
        'logOR_train': logOR_train,
        'logOR_test': logOR_test,
        'F1_test': f1_test
    }

    if bool(IIDs_nodisease):
        data_test_pancohort = data[data.index.isin(testIIDs+IIDs_nodisease)]
        pred_probs_test_pancohort = pd.DataFrame({"predicted": lasso_model.predict_proba(data_test_pancohort[cols_X])[:, 1]}, index=data_test_pancohort.index)
        logOR_test_pancohort = logOR(data_test_pancohort, col_y, pred_probs_test_pancohort)
        res['pred_probs_test_pancohort'] = pred_probs_test_pancohort
        res['logOR_test_pancohort'] = logOR_test_pancohort

    return res

def fit_lasso(data, trainIIDs, testIIDs, prs_cols, IIDs_nodisease=[], col_y='BinCAT_Disease', cont_covars=[], cat_covars=[], use_scaler=False, max_iter=1000, random_state=42, threads=5, mode="lasso"):
    data_train = data[data.index.isin(trainIIDs)]
    data_test = data[data.index.isin(testIIDs)]

    cols_X = prs_cols.copy()

    if cont_covars:
        cols_X += cont_covars

    transformers = []
    if use_scaler:
        transformers.append(('num', StandardScaler(), prs_cols+cont_covars))
    else:
        transformers.append(('num', FunctionTransformer(accept_sparse=True, check_inverse=False, feature_names_out='one-to-one'), prs_cols+cont_covars))

    if cat_covars:
        cols_X += cat_covars  
        transformers.append(('cat', OneHotEncoder(), cat_covars))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough' 
    )

    if mode == "Lasso":
        lasso_model = LogisticRegressionCV(cv=5, penalty='l1', solver='saga', max_iter=max_iter, random_state=random_state, n_jobs=threads)
        step_name = 'logisticregressioncv'
    elif "ToLassoORnotToLasso" in mode:
        idx_lasso_features = np.arange(len(prs_cols))
        idx_non_lasso_features = np.arange(len(prs_cols), len(prs_cols) + len(cont_covars) + len(OneHotEncoder().fit(data_train[cat_covars]).get_feature_names_out(cat_covars)))
        lasso_model = ToLassoORnotToLasso(
                            idx_lasso_features=idx_lasso_features,
                            idx_non_lasso_features=idx_non_lasso_features,
                            lasso_params={'cv': 5, 'max_iter': max_iter, 'random_state': random_state, 'n_jobs': threads},
                            non_lasso_nonCV=True if mode in ["ToLassoORnotToLasso_simple", "ToLassoORnotToLasso"] else False, 
                            logistic_params={'random_state': random_state, 'n_jobs': threads}
                        )
        step_name = 'tolassoornottolasso'
    elif "LassoSteps" in mode:
        idx_lasso_features = np.arange(len(prs_cols))
        idx_non_lasso_features = np.arange(len(prs_cols), len(prs_cols) + len(cont_covars) + len(OneHotEncoder().fit(data_train[cat_covars]).get_feature_names_out(cat_covars)))
        lasso_model = LassoSteps(
                            idx_lasso_features=idx_lasso_features,
                            idx_non_lasso_features=idx_non_lasso_features,
                            lasso_params={'cv': 5, 'max_iter': 10, 'random_state': random_state, 'n_jobs': threads},
                            logistic_params={'random_state': random_state, 'n_jobs': threads},
                            mode=int(mode.split("LassoSteps")[-1])
                        )
        step_name = 'lassosteps'
    elif mode == "GroupLasso":
        group_ids = [1] * len(prs_cols) + [0] * (len(cont_covars) + len(OneHotEncoder().fit(data_train[cat_covars]).get_feature_names_out(cat_covars)))
        group_lasso = LogisticGroupLasso(
                        groups=group_ids,
                        group_reg=np.array([0.1, 1e-10]),  # reg param for the first group will be tuned by GridSearchCV, second group will be fixed with a very small reg 
                        l1_reg=0.1,    # This will be tuned by GridSearchCV
                        scale_reg="group_size",  # Scale regularisation by group size
                        subsampling_scheme=None,
                        supress_warning=True,
                        n_iter=max_iter,
                        tol=1e-3,
                        random_state=random_state,
                        fit_intercept=True
                    )
        param_grid = {
                        'group_reg': [np.array([alpha, 1e-10]) for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]],
                        'l1_reg': [0.0, 0.01, 0.1, 0.5, 1.0]
                    }
        lasso_model = GridSearchCV(group_lasso, param_grid, cv=5, scoring="accuracy", n_jobs=threads)
    elif mode == "ElasticNet":
        lasso_model = LogisticRegressionCV(cv=5, penalty='elasticnet', solver='saga', max_iter=max_iter, random_state=random_state, n_jobs=threads, l1_ratios=[0.5])
        step_name = 'logisticregressioncv'
    elif mode == "XGBoost":
        from xgboost import XGBClassifier
        lasso_model = XGBClassifier(booster='dart', 
                                    reg_alpha=1, # L1 regularization term on weights
                                    reg_lambda=1, # L2 regularization term on weights
                                    objective='binary:logistic', 
                                    n_estimators=50, 
                                    use_label_encoder=False, eval_metric='aucpr', random_state=random_state, n_jobs=threads)
        
    lasso_model = make_pipeline(preprocessor, lasso_model)

    lasso_model.fit(data_train[cols_X], data_train[col_y])

    if mode == "GroupLasso":
        best_gl = lasso_model.named_steps['gridsearchcv'].best_estimator_
        lasso_model = make_pipeline(preprocessor, best_gl)
        lasso_model.fit(data_train[cols_X], data_train[col_y])

    pred_probs_train = pd.DataFrame({"predicted": lasso_model.predict_proba(data_train[cols_X])[:, 1]}, index=data_train.index)
    pred_probs_test = pd.DataFrame({"predicted": lasso_model.predict_proba(data_test[cols_X])[:, 1]}, index=data_test.index)
    auc_train = roc_auc_score(data_train[col_y], pred_probs_train)
    logOR_train = logOR(data_train, col_y, pred_probs_train)
    auc_test = roc_auc_score(data_test[col_y], pred_probs_test)
    logOR_test = logOR(data_test, col_y, pred_probs_test)

    f1_test = f1_score(data_test[col_y], lasso_model.predict(data_test[cols_X]))

    features = lasso_model.named_steps['columntransformer'].get_feature_names_out()
    features = [f.split("__")[1] for f in features]
    if mode == "GroupLasso":
        coef_results = pd.DataFrame({'Feature': features, 'isSelected': lasso_model.named_steps['logisticgrouplasso'].sparsity_mask})
    else:
        coef_results = pd.DataFrame({'Feature': features, 'Coefficient': lasso_model.named_steps[step_name].coef_[0]}) if (mode != "XGBoost") and not ("LassoSteps" in mode) else [-1]
    
    res = {
        'Coeff': coef_results,
        'pred_probs_train': pred_probs_train,
        'pred_probs_test': pred_probs_test,
        'AUC_train': auc_train,
        'AUC_test': auc_test,
        'logOR_train': logOR_train,
        'logOR_test': logOR_test,
        'F1_test': f1_test
    }

    if bool(IIDs_nodisease):
        data_test_pancohort = data[data.index.isin(testIIDs+IIDs_nodisease)]
        pred_probs_test_pancohort = pd.DataFrame({"predicted": lasso_model.predict_proba(data_test_pancohort[cols_X])[:, 1]}, index=data_test_pancohort.index)
        logOR_test_pancohort = logOR(data_test_pancohort, col_y, pred_probs_test_pancohort)
        res['pred_probs_test_pancohort'] = pred_probs_test_pancohort
        res['logOR_test_pancohort'] = logOR_test_pancohort

    return res

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

def getF1(label, pred_prob, method="at50"):
    match method:
        case "otsu":        
            threshold = threshold_otsu(pred_prob.values)
        case "at50":
            threshold = 0.5
        case _:
            sys.exit(f"Method {method} for getF1 not implemented")
    pred = pred_prob > threshold
    return f1_score(label, pred)

def eval_glm(results, noP=False):    
    resDF = pd.DataFrame(results.tolist(), index=results.index)
    resDF.sort_values(by='AUC_test', ascending=False, inplace=True)
    return resDF

# %%
def getARGSParser():
    parser = argparse.ArgumentParser(description='MultiPRS Script')
    parser.add_argument('--prs_res_root', type=str, help='Path to PRS results root directory', default="/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc")
    parser.add_argument('--rds_pres_prefix', type=str, help='Prefix before the pheno name in the RDS file name', default="run_ext_basic_king0p0625_lw_gw_indep_FiltMAF_")
    parser.add_argument('--rds_pres_suffix', type=str, help='Suffix after the pheno name in the RDS file name', default=".fullDS.auto.mod.LDPred2.rds")
    parser.add_argument('--rds_tag_prs', type=str, help='tag PRS present in the rds file name', default="auto.mod")
    parser.add_argument('--tag_data', type=str, help='tag PRS model', default="resNdata.basic")
    parser.add_argument('--tag_prs', type=str, help='tag PRS inside the rds file', default="pred_auto")

    parser.add_argument('--disease_csv', type=str, help='Path to disease CSV file', default="/project/ukbblatent/clinicaldata/binary_disease_cohorts/F20208v3_nonDiscov/caucasian_king0p0625_grouped/newcovsets/V1/hypertension.csv")
    parser.add_argument('--col_disease', type=str, default='BinCAT_Disease')
    parser.add_argument('--min_sub', type=int, help='Minimum number of subjects must be present in the dataset. Set it to 0 or None to ignore this filter.', default=1000)

    parser.add_argument('--output_root', type=str, help='Path to store the output', default="/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/newcovsets_V1/caucasian_king0p0625_grouped/panCohort_auto_lw_gw_10kIT_kingB4ldpred2")

    parser.add_argument('--ext_covar', type=str, help='External covariates file is to be used, in addition to the fulldata file of the PRS. Keep it blank to only use the fulldata.', default="/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/covars/nonDisc_caucasian_king0p0625_V1.tsv")
    parser.add_argument('--cont_covar_cols', type=str, help='Comma-separated list of continuous covariate columns', default='BMI,Systolic_blood_pressure,Diastolic_blood_pressure,LDL_Cholesterol')
    parser.add_argument('--nPCs_covar', type=int, help='Number of principal components for covariates', default=0)
    parser.add_argument('--cat_covar_cols', type=str, help='Comma-separated list of categorical covariate columns', default='CAT_Smoking')
    parser.add_argument('--lassoCV_max_iter', type=int, help='Maximum number of iterations for LassoCV', default=100)
    parser.add_argument('--mode_multi', type=str, help='Whether to use Lasso of ToLassoORnotToLasso_simple (or ToLassoORnotToLasso_CV) or GroupLasso or ElasticNet (or XGBosst) for the multi-models', default="LassoSteps3")
    parser.add_argument('--threads', type=int, help='Number of threads', default=5)

    parser.add_argument('--do_singlePRS', action=argparse.BooleanOptionalAction, default=False, help='Run single PRS models')
    parser.add_argument('--do_singlePRSCovar', action=argparse.BooleanOptionalAction, default=False, help='Run single PRS + Covariate models')
    parser.add_argument('--do_singlePRSCovarNorm', action=argparse.BooleanOptionalAction, default=False, help='Run single PRS + Covariate models with normalisation')
    parser.add_argument('--do_covar', action=argparse.BooleanOptionalAction, default=False, help='Run Covariate models')
    parser.add_argument('--do_covarNorm', action=argparse.BooleanOptionalAction, default=False, help='Run Covariate models with normalisation')
    parser.add_argument('--do_nonPCCovar', action=argparse.BooleanOptionalAction, default=False, help='Whether to additioanlly run covar-related PRS models without PCs as covariates (Only if nPCs_covar > 0)')

    parser.add_argument('--do_multiPRS', action=argparse.BooleanOptionalAction, default=False, help='Run multiPRS models')
    parser.add_argument('--do_multiPRSNorm', action=argparse.BooleanOptionalAction, default=False, help='Run multi normalised PRS models')
    parser.add_argument('--do_multiPRSCovar', action=argparse.BooleanOptionalAction, default=True, help='Run multiPRS + Covariate models')
    parser.add_argument('--do_multiPRSNormCovar', action=argparse.BooleanOptionalAction, default=True, help='Run multi normalised PRS + Covariate models')

    return parser

if __name__ == "__main__":
    parser = getARGSParser()
    args, _ = parser.parse_known_args()
    print(f"Multi mode: {args.mode_multi}...................................")

    # %%
    os.makedirs(args.output_root, exist_ok=True)
    print(f"DisCSV: {args.disease_csv}")

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
    # Process the data for the subjects that are not part of the disease cohort

    Xy_nodisease = fulldata[~fulldata.IID.isin(Xy.index)]
    Xy_nodisease[args.col_disease] = 0
    Xy_nodisease.set_index("IID", inplace=True)
    IIDs_nodisease = list(set(Xy_nodisease.index))

    prs_Xy_nodisease = Xy_nodisease.filter(regex='^PRS|'+args.col_disease)
    prs_Xy_nodisease = prs_Xy_nodisease.reset_index().melt(id_vars=['IID', args.col_disease], var_name='PRS_Type', value_name='PRS_Score')
    prs_Xy_nodisease.set_index('IID', inplace=True)

    prs_Xy_covar_nodisease = prs_Xy_nodisease.join(covars)

    list(Xy_nodisease.index)

    # %%
    # Merge disease and nodisease data

    Xy = pd.concat([Xy, Xy_nodisease])
    prs_Xy = pd.concat([prs_Xy, prs_Xy_nodisease])
    prs_Xy_covar = pd.concat([prs_Xy_covar, prs_Xy_covar_nodisease])

    # %%
      
    kf = KFold(n_splits=5, shuffle=True, random_state=1701)

    res_store = defaultdict(recursive_defaultdict)

    for fold, (train_index, test_index) in tqdm(enumerate(kf.split(IIDs), 1)):
        trainIIDs = [IIDs[index] for index in train_index]
        testIIDs = [IIDs[index] for index in test_index]
        print(f"Fold {fold}")

        res_store[f"Fold_{fold}"]['IDs']['train'] = trainIIDs
        res_store[f"Fold_{fold}"]['IDs']['test'] = testIIDs

        #############################

        #Baseline models

        #Predict the disease from the covariates (with and without normalisation)
        if args.do_covar:
            res_store[f"Fold_{fold}"]['GLM']['covar'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, x_var='', col_y=args.col_disease, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols)            
            # res_store[f"Fold_{fold}"][args.mode_multi]['covar'] = fit_lasso(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=[], cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=False, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi)

            #without PC covars
            if args.do_nonPCCovar and bool(args.nPCs_covar):
                res_store[f"Fold_{fold}"]['GLM']['nonPCCovar'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, x_var='', col_y=args.col_disease, cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols)
                # res_store[f"Fold_{fold}"][args.mode_multi]['nonPCCovar'] = fit_lasso(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=[], cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=False, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi)

        if args.do_covarNorm:
            res_store[f"Fold_{fold}"]['GLM']['covarNorm'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, x_var='', col_y=args.col_disease, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols, use_scaler=True)
            # res_store[f"Fold_{fold}"][args.mode_multi]['covarNorm'] = fit_lasso(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=[], cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=True, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi)

            if args.do_nonPCCovar and bool(args.nPCs_covar):
                res_store[f"Fold_{fold}"]['GLM']['nonPCCovarNorm'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, x_var='', col_y=args.col_disease, cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols, use_scaler=True)
                # res_store[f"Fold_{fold}"][args.mode_multi]['nonPCCovarNorm'] = fit_lasso(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=[], cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=True, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi)

        ###############################

        #Single PRS models

        #Predict the disease from individual PRS
        if args.do_singlePRS:
            results = prs_Xy.groupby('PRS_Type').apply(lambda x: fit_logistic_regression(x, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, col_y=args.col_disease))
            res_store[f"Fold_{fold}"]['GLM']['singlePRS'] = eval_glm(results)

        #Predict the disase from individual PRS and covariates (with and without normalisation)
        if args.do_singlePRSCovar:
            results = prs_Xy_covar.groupby('PRS_Type').apply(lambda x: fit_logistic_regression(x, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, col_y=args.col_disease, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols))
            res_store[f"Fold_{fold}"]['GLM']['singlePRSCovar'] = eval_glm(results)

            # results = prs_Xy_covar.groupby('PRS_Type').apply(lambda x: fit_lasso(x, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=['PRS_Score'], col_y=args.col_disease, use_scaler=False, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi))
            # res_store[f"Fold_{fold}"][args.mode_multi]['singlePRSCovar'] = eval_glm(results, noP=True)

        if args.do_singlePRSCovarNorm:
            results = prs_Xy_covar.groupby('PRS_Type').apply(lambda x: fit_logistic_regression(x, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, col_y=args.col_disease, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols, use_scaler=True))
            res_store[f"Fold_{fold}"]['GLM']['singlePRSCovarNorm'] = eval_glm(results)

            # results = prs_Xy_covar.groupby('PRS_Type').apply(lambda x: fit_lasso(x, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=['PRS_Score'], col_y=args.col_disease, use_scaler=True, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi))
            # res_store[f"Fold_{fold}"][args.mode_multi]['singlePRSCovarNorm'] = eval_glm(results, noP=True)
        
        ################################
        #MultiPRS models

        #multiPRS
        if args.do_multiPRS:
            # res_store[f"Fold_{fold}"]['GLM']['multiPRS'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, x_var='', prs_cols=prs_cols, col_y=args.col_disease, use_scaler=False)
            res_store[f"Fold_{fold}"][args.mode_multi]['multiPRS'] = fit_lasso(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=prs_cols, col_y=args.col_disease, use_scaler=False, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi)

        #multiPRS normalised
        if args.do_multiPRSNorm:
            # res_store[f"Fold_{fold}"]['GLM']['multiPRSNorm'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, x_var='', prs_cols=prs_cols, col_y=args.col_disease, use_scaler=True)
            res_store[f"Fold_{fold}"][args.mode_multi]['multiPRSNorm'] = fit_lasso(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=prs_cols, col_y=args.col_disease, use_scaler=True, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi)

        #multiPRS + covar
        if args.do_multiPRSCovar:
            # res_store[f"Fold_{fold}"]['GLM']['multiPRSCovar'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, x_var='', prs_cols=prs_cols, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=False)
            res_store[f"Fold_{fold}"][args.mode_multi]['multiPRSCovar'] = fit_lasso(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=prs_cols, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=False, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi)
            
            #without PC covars
            if args.do_nonPCCovar and bool(args.nPCs_covar):
                # res_store[f"Fold_{fold}"]['GLM']['multiPRSnonPCCovar'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, x_var='', prs_cols=prs_cols, cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=False)
                res_store[f"Fold_{fold}"][args.mode_multi]['multiPRSnonPCCovar'] = fit_lasso(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=prs_cols, cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=False, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi)

        #multiPRS + covar
        if args.do_multiPRSNormCovar:
            # res_store[f"Fold_{fold}"]['GLM']['multiPRSNormCovar'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, x_var='', prs_cols=prs_cols, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=True)
            res_store[f"Fold_{fold}"][args.mode_multi]['multiPRSNormCovar'] = fit_lasso(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=prs_cols, cont_covars=args.cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=True, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi)
            
            #without PC covars
            if args.do_nonPCCovar and bool(args.nPCs_covar):
                # res_store[f"Fold_{fold}"]['GLM']['multiPRSNormnonPCCovar'] = fit_logistic_regression(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, x_var='', prs_cols=prs_cols, cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=True)
                res_store[f"Fold_{fold}"][args.mode_multi]['multiPRSNormnonPCCovar'] = fit_lasso(Xy, trainIIDs, testIIDs, IIDs_nodisease=IIDs_nodisease, prs_cols=prs_cols, cont_covars=nonPC_cont_covar_cols, cat_covars=args.cat_covar_cols, col_y=args.col_disease, use_scaler=True, max_iter=args.lassoCV_max_iter, random_state=42, threads=args.threads, mode=args.mode_multi)
                

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
                        "logOR_test": res_store[fold][method][res_type]['logOR_test'],
                        "F1_test": res_store[fold][method][res_type]['F1_test']
                    }
                    if bool(IIDs_nodisease):
                        datum["logOR_test_pancohort"] = res_store[fold][method][res_type]['logOR_test_pancohort']
                else:
                    datum = {
                        "fold": fold.replace("Fold_", ""),
                        "method": method,
                        "res_type": res_type,
                        "AUC_test": res_store[fold][method][res_type]['AUC_test'].max(),
                        "logOR_test": res_store[fold][method][res_type]['logOR_test'].max(),
                        "F1_test": res_store[fold][method][res_type]['F1_test'].max()
                    }
                    if bool(IIDs_nodisease):
                        datum["logOR_test_pancohort"] = res_store[fold][method][res_type]['logOR_test_pancohort'].max()
                score_store.append(datum)

    score_store = pd.DataFrame(score_store)
    score_store.to_csv(f"{args.output_root}/{args.tag_disease}_models_test_scores.tsv", sep="\t", index=False)    
    print(f"{args.output_root}/{args.tag_disease}_models_test_scores.tsv")

