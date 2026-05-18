# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
import os
import argparse

# sys.path.insert(0, "/group/glastonbury/soumick/MyCodes/GitLab/tricorder")
# from utils.python_utils import recursive_defaultdict,setup_pandas_compatibility
# setup_pandas_compatibility()

# import pickle

import pyreadr
from rds2py import read_rds

import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional

# Optional Bayesian Dependencies
try:
    import pymc as pm
    import arviz as az
    BAYES_AVAILABLE = True
except ImportError:
    BAYES_AVAILABLE = False
    print("Warning: 'pymc' not found. Bayesian Hierarchical strategy will be skipped.")

try:
    import bambi as bmb
    import arviz as az
    BAMBI_AVAILABLE = True
except ImportError:
    BAMBI_AVAILABLE = False
    print("Warning: 'bambi' or 'arviz' not found. Bayesian strategies will be skipped.")

class PRSEvaluator:

    def __init__(self, covar_df: pd.DataFrame, prs_df: pd.DataFrame,
                 sota_df: pd.DataFrame, disease_df: pd.DataFrame,
                 target_col: str = "BinCAT_Disease",
                 covariates: List[str] = None,
                 cat_covariates: List[str] = None,
                 sota_col: str = "prs_sota_CVD",
                 sota_label: str = "SOTA Benchmark",
                 orthogonalise: bool = False,
                 output_dir: str = None,
                 use_variational_inference: bool = False,
                 run_best_single: bool = True,
                 run_elastic_net: bool = True,
                 run_pca: bool = True,
                 run_bayesian_pymc: bool = True,
                 run_bayesian_bambi: bool = True,
                 run_with_sota: bool = False):
        print("Initialising PRS Evaluator")
        self.raw_data = covar_df.join([prs_df, sota_df, disease_df], how='inner')

        initial_n = len(self.raw_data)
        self.data = self.raw_data.dropna()
        print(f"Data merged. N={len(self.data)} (Dropped {initial_n - len(self.data)} rows with missing data).")

        self.target_col = target_col
        self.covariates = covariates if covariates is not None else ["Age", "Sex", "BMI"]
        self.cat_covariates = cat_covariates if cat_covariates is not None else ["CAT_Smoking"]
        self.sota_col = sota_col
        self.sota_label = sota_label
        self.my_prs_cols = list(prs_df.columns)
        self.use_variational_inference = use_variational_inference
        
        self.run_best_single = run_best_single
        self.run_elastic_net = run_elastic_net
        self.run_pca = run_pca
        self.run_bayesian_pymc = run_bayesian_pymc
        self.run_bayesian_bambi = run_bayesian_bambi
        self.run_with_sota = run_with_sota
        
        self.output_dir = output_dir
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {self.output_dir}")

        self.data = pd.get_dummies(self.data, columns=self.cat_covariates, drop_first=True)
        new_covars = [c for c in self.data.columns if "CAT_Smoking" in c]
        self.final_covariates = self.covariates + new_covars

        if orthogonalise:
            self._orthogonalise_prs_features()

        self._preprocess_data()

    def _orthogonalise_prs_features(self):
        """Residualise PRS features against covariates."""
        print("Performing feature orthogonalisation (residualisation)")
        X_nuisance = self.data[self.final_covariates].astype(float)
        all_prs_cols = self.my_prs_cols + [self.sota_col]
        lr = LinearRegression()
        for col in all_prs_cols:
            lr.fit(X_nuisance, self.data[col])
            self.data[col] = self.data[col] - lr.predict(X_nuisance)
        print(f"PRS features residualised against {len(self.final_covariates)} covariates.")

    def _preprocess_data(self):
        scaler = StandardScaler()
        continuous_covars = [c for c in self.covariates if c in self.data.columns]
        cols_to_scale = continuous_covars + [self.sota_col] + self.my_prs_cols
        self.data[cols_to_scale] = scaler.fit_transform(self.data[cols_to_scale])
        print("Standardisation complete: Z-scores calculated for continuous variables.")

    def _delong_test(self, preds_a, preds_b, target):
        """DeLong's test p-value for correlated ROC curves (Hanley-McNeil approximation)."""
        y_true = np.array(target).astype(int)

        if len(y_true) == 0 or len(np.unique(y_true)) < 2: 
            return np.nan
            
        try:
            auc_a = roc_auc_score(y_true, preds_a)
            auc_b = roc_auc_score(y_true, preds_b)
        except ValueError:
            return np.nan

        r = np.corrcoef(preds_a, preds_b)[0, 1]
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return np.nan
            
        se_a = np.sqrt(auc_a * (1 - auc_a) / n_pos + auc_a * (1 - auc_a) / n_neg)
        se_b = np.sqrt(auc_b * (1 - auc_b) / n_pos + auc_b * (1 - auc_b) / n_neg)

        denom = np.sqrt(se_a**2 + se_b**2 - 2*r*se_a*se_b)
        if denom == 0 or np.isnan(denom): 
            return 1.0

        z = (auc_a - auc_b) / denom
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        return p_val

    def evaluate_single_prs(self, prs_col_name: str) -> Dict:
        X = self.data[self.final_covariates + [prs_col_name]].astype(float)
        y = self.data[self.target_col]

        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]

        return {
            "Name": prs_col_name,
            "AUC": roc_auc_score(y, probs),
            "Probs": probs,
            "OR_per_SD": np.exp(model.coef_[0][-1])
        }

    def find_optimal_combination_elastic_net(self) -> Dict:
        print("\nStrategy A: Elastic Net")
        X = self.data[self.final_covariates + self.my_prs_cols].astype(float)
        y = self.data[self.target_col]

        el_net = LogisticRegressionCV(
            cv=5, penalty='elasticnet', solver='saga',
            l1_ratios=[0.1, 0.5, 0.7, 0.95], max_iter=5000, n_jobs=-1, random_state=42
        )
        el_net.fit(X, y)
        probs = el_net.predict_proba(X)[:, 1]

        coefs = el_net.coef_[0][len(self.final_covariates):]
        n_selected = np.sum(coefs != 0)
        print(f"Elastic Net retained {n_selected} PRS features.")

        return {
            "Name": "Composite (ElasticNet)",
            "AUC": roc_auc_score(y, probs),
            "Probs": probs
        }

    def find_optimal_combination_elastic_net_with_sota(self) -> Dict:
        print("\nStrategy A+: Elastic Net with SOTA PRS")
        X = self.data[self.final_covariates + self.my_prs_cols + [self.sota_col]].astype(float)
        y = self.data[self.target_col]

        el_net = LogisticRegressionCV(
            cv=5, penalty='elasticnet', solver='saga',
            l1_ratios=[0.1, 0.5, 0.7, 0.95], max_iter=5000, n_jobs=-1, random_state=42
        )
        el_net.fit(X, y)
        probs = el_net.predict_proba(X)[:, 1]

        # Count non-zero PRS coefficients
        coefs = el_net.coef_[0][len(self.final_covariates):]
        n_selected = np.sum(coefs != 0)
        print(f"Elastic Net (with SOTA) retained {n_selected} PRS features.")

        return {
            "Name": "Composite (ElasticNet + SOTA)",
            "AUC": roc_auc_score(y, probs),
            "Probs": probs
        }

    def find_optimal_combination_pca(self, variance_threshold: float = 0.95) -> Dict:
        print("\nStrategy B: PCLR")

        X_prs = self.data[self.my_prs_cols].astype(float)

        pca = PCA(n_components=variance_threshold)
        X_pca = pca.fit_transform(X_prs)

        n_components = X_pca.shape[1]
        print(f"PCA reduced {len(self.my_prs_cols)} PRS traits to {n_components} orthogonal components.")

        pca_cols = [f"PC_{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=self.data.index)

        X_final = pd.concat([self.data[self.final_covariates].astype(float), df_pca], axis=1)
        y = self.data[self.target_col]

        model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
        model.fit(X_final, y)

        probs = model.predict_proba(X_final)[:, 1]

        return {
            "Name": "Composite (PCA-95%)",
            "AUC": roc_auc_score(y, probs),
            "Probs": probs
        }

    def find_optimal_combination_pca_with_sota(self, variance_threshold: float = 0.95) -> Dict:
        print("\nStrategy B+: PCLR with SOTA PRS")

        X_prs = self.data[self.my_prs_cols].astype(float)

        pca = PCA(n_components=variance_threshold)
        X_pca = pca.fit_transform(X_prs)

        n_components = X_pca.shape[1]
        print(f"PCA reduced {len(self.my_prs_cols)} PRS traits to {n_components} orthogonal components.")

        pca_cols = [f"PC_{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=self.data.index)

        X_final = pd.concat([self.data[self.final_covariates].astype(float), df_pca, self.data[[self.sota_col]].astype(float)], axis=1)
        y = self.data[self.target_col]

        model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
        model.fit(X_final, y)

        probs = model.predict_proba(X_final)[:, 1]

        return {
            "Name": "Composite (PCA-95% + SOTA)",
            "AUC": roc_auc_score(y, probs),
            "Probs": probs
        }

    def find_optimal_combination_bayesian(self) -> Dict:
        """Bayesian hierarchical logistic regression via PyMC."""
        if not BAYES_AVAILABLE:
            return {"Name": "Bayesian (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}

        print("\nStrategy C: Bayesian Hierarchical Regression (PyMC)")

        try:
            X_cov = self.data[self.final_covariates].astype(float).values
            X_prs = self.data[self.my_prs_cols].astype(float).values
            y_obs = self.data[self.target_col].values
        except ValueError as e:
            print(f"[ERROR] Data conversion for Bayesian model failed: {e}")
            return {"Name": "Bayesian (Failed)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}        

        n_cov = X_cov.shape[1]
        n_prs = X_prs.shape[1]

        with pm.Model() as hierarchical_model:
            beta_cov = pm.Normal("beta_cov", mu=0, sigma=2, shape=n_cov)

            tau = pm.HalfNormal("tau", sigma=1)
            beta_prs = pm.Normal("beta_prs", mu=0, sigma=tau, shape=n_prs)

            alpha = pm.Normal("alpha", mu=0, sigma=2)

            mu = alpha + pm.math.dot(X_cov, beta_cov) + pm.math.dot(X_prs, beta_prs)

            y_est = pm.Bernoulli("y_est", logit_p=mu, observed=y_obs)

            if self.use_variational_inference:
                print("Using Variational Inference (ADVI)...")
                inference = pm.ADVI()
                approx = pm.fit(n=20000, method=inference, progressbar=True)
                trace = approx.sample(2000)
                posterior_alpha = float(trace.posterior["alpha"].mean().values)
                posterior_cov = trace.posterior["beta_cov"].mean(dim="draw").values.flatten()
                posterior_prs = trace.posterior["beta_prs"].mean(dim="draw").values.flatten()
            else:
                print("Using MCMC (NUTS)...")
                trace = pm.sample(draws=1000, tune=1000, target_accept=0.9, chains=4, cores=4, progressbar=True)
                posterior_alpha = float(trace.posterior["alpha"].mean(dim=["chain", "draw"]).values)
                posterior_cov = trace.posterior["beta_cov"].mean(dim=["chain", "draw"]).values
                posterior_prs = trace.posterior["beta_prs"].mean(dim=["chain", "draw"]).values

            logit_p = posterior_alpha + np.dot(X_cov, posterior_cov) + np.dot(X_prs, posterior_prs)
            probs = 1 / (1 + np.exp(-logit_p))

        inference_method = "VI-ADVI" if self.use_variational_inference else "MCMC-NUTS"
        print(f"Bayesian Sampling Complete ({inference_method}).")
        auc = roc_auc_score(y_obs, probs)

        return {
            "Name": "Composite (Bayesian Hierarchical)",
            "AUC": auc,
            "Probs": probs
        }

    def find_optimal_combination_bayesian_with_sota(self) -> Dict:
        """Bayesian hierarchical logistic regression with SOTA PRS via PyMC."""
        if not BAYES_AVAILABLE:
            return {"Name": "Bayesian + SOTA (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}

        print("\nStrategy C+: Bayesian Hierarchical Regression with SOTA (PyMC)")

        try:
            X_cov = self.data[self.final_covariates].astype(float).values
            X_prs = self.data[self.my_prs_cols].astype(float).values
            X_sota = self.data[self.sota_col].astype(float).values.reshape(-1, 1)
            y_obs = self.data[self.target_col].values
        except ValueError as e:
            print(f"[ERROR] Data conversion for Bayesian model failed: {e}")
            return {"Name": "Bayesian + SOTA (Failed)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}

        n_cov = X_cov.shape[1]
        n_prs = X_prs.shape[1]

        with pm.Model() as hierarchical_model:
            beta_cov = pm.Normal("beta_cov", mu=0, sigma=2, shape=n_cov)

            tau = pm.HalfNormal("tau", sigma=1)
            beta_prs = pm.Normal("beta_prs", mu=0, sigma=tau, shape=n_prs)

            beta_sota = pm.Normal("beta_sota", mu=0, sigma=2)

            alpha = pm.Normal("alpha", mu=0, sigma=2)

            mu = alpha + pm.math.dot(X_cov, beta_cov) + pm.math.dot(X_prs, beta_prs) + X_sota.flatten() * beta_sota

            y_est = pm.Bernoulli("y_est", logit_p=mu, observed=y_obs)

            if self.use_variational_inference:
                print("Using Variational Inference (ADVI)...")
                inference = pm.ADVI()
                approx = pm.fit(n=20000, method=inference, progressbar=True)
                trace = approx.sample(2000)
                posterior_alpha = float(trace.posterior["alpha"].mean().values)
                posterior_cov = trace.posterior["beta_cov"].mean(dim="draw").values.flatten()
                posterior_prs = trace.posterior["beta_prs"].mean(dim="draw").values.flatten()
                posterior_sota = float(trace.posterior["beta_sota"].mean().values)
            else:
                print("Using MCMC (NUTS)...")
                trace = pm.sample(draws=1000, tune=1000, target_accept=0.9, chains=4, cores=4, progressbar=True)
                posterior_alpha = float(trace.posterior["alpha"].mean(dim=["chain", "draw"]).values)
                posterior_cov = trace.posterior["beta_cov"].mean(dim=["chain", "draw"]).values
                posterior_prs = trace.posterior["beta_prs"].mean(dim=["chain", "draw"]).values
                posterior_sota = float(trace.posterior["beta_sota"].mean(dim=["chain", "draw"]).values)

            logit_p = posterior_alpha + np.dot(X_cov, posterior_cov) + np.dot(X_prs, posterior_prs) + X_sota.flatten() * posterior_sota
            probs = 1 / (1 + np.exp(-logit_p))

        inference_method = "VI-ADVI" if self.use_variational_inference else "MCMC-NUTS"
        print(f"Bayesian Sampling Complete ({inference_method}).")
        auc = roc_auc_score(y_obs, probs)

        return {
            "Name": "Composite (Bayesian Hierarchical + SOTA)",
            "AUC": auc,
            "Probs": probs
        }

    def find_optimal_combination_bambi(self) -> Dict:
        """Bayesian hierarchical logistic regression via Bambi."""
        if not BAMBI_AVAILABLE:
            return {"Name": "Bambi (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}

        print("\nStrategy C: Bayesian Hierarchical Regression (Bambi)")

        prs_formula_part = " + ".join([f"`{col}`" for col in self.my_prs_cols])
        cov_formula_part = " + ".join([f"`{col}`" for col in self.final_covariates])
        formula = f"`{self.target_col}` ~ {cov_formula_part} + {prs_formula_part}"
        
        nested_sigma = bmb.Prior("HalfNormal", sigma=1)
        hierarchical_prior = bmb.Prior("Normal", mu=0, sigma=nested_sigma)
        my_priors = {f"`{prs_col}`": hierarchical_prior for prs_col in self.my_prs_cols}

        model = bmb.Model(formula, data=self.data, family="bernoulli", priors=my_priors)
        
        if self.use_variational_inference:
            print("Using Variational Inference (ADVI)...")
            approx = model.fit(method="vi", inference_method="advi", n=20000, random_seed=42, progressbar=True)
            inference_method = "VI-ADVI"
            trace = approx.sample(2000)
            model.predict(trace, kind="response", inplace=True)
            probs = trace.posterior_predictive[self.target_col].mean(dim="draw").values.flatten()
        else:
            print("Using MCMC (NUTS)...")
            results = model.fit(draws=1000, tune=1000, target_accept=0.9, chains=4, cores=4, random_seed=42, progressbar=True)
            inference_method = "MCMC-NUTS"
            model.predict(results, kind="response")
            probs = results.posterior[f"{self.target_col}_mean"].mean(dim=("chain", "draw")).values

        print(f"Bambi Sampling Complete ({inference_method}).")
        auc = roc_auc_score(self.data[self.target_col], probs)
        
        return {
            "Name": "Composite (Bambi Hierarchical)",
            "AUC": auc,
            "Probs": probs
        }

    def find_optimal_combination_bambi_with_sota(self) -> Dict:
        """Bayesian hierarchical logistic regression with SOTA PRS via Bambi."""
        if not BAMBI_AVAILABLE:
            return {"Name": "Bambi + SOTA (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}

        print("\nStrategy D+: Bayesian Hierarchical Regression with SOTA (Bambi)")

        prs_formula_part = " + ".join([f"`{col}`" for col in self.my_prs_cols])
        cov_formula_part = " + ".join([f"`{col}`" for col in self.final_covariates])
        formula = f"`{self.target_col}` ~ {cov_formula_part} + {prs_formula_part} + `{self.sota_col}`"
        
        nested_sigma = bmb.Prior("HalfNormal", sigma=1)
        hierarchical_prior = bmb.Prior("Normal", mu=0, sigma=nested_sigma)
        my_priors = {f"`{prs_col}`": hierarchical_prior for prs_col in self.my_prs_cols}
        my_priors[f"`{self.sota_col}`"] = bmb.Prior("Normal", mu=0, sigma=2)

        model = bmb.Model(formula, data=self.data, family="bernoulli", priors=my_priors)
        
        if self.use_variational_inference:
            print("Using Variational Inference (ADVI)...")
            approx = model.fit(method="vi", inference_method="advi", n=20000, random_seed=42, progressbar=True)
            inference_method = "VI-ADVI"
            trace = approx.sample(2000)
            model.predict(trace, kind="response", inplace=True)
            probs = trace.posterior_predictive[self.target_col].mean(dim="draw").values.flatten()
        else:
            print("Using MCMC (NUTS)...")
            results = model.fit(draws=1000, tune=1000, target_accept=0.9, chains=4, cores=4, random_seed=42, progressbar=True)
            inference_method = "MCMC-NUTS"
            model.predict(results, kind="response")
            probs = results.posterior[f"{self.target_col}_mean"].mean(dim=("chain", "draw")).values

        print(f"Bambi Sampling Complete ({inference_method}).")
        auc = roc_auc_score(self.data[self.target_col], probs)
        
        return {
            "Name": "Composite (Bambi Hierarchical + SOTA)",
            "AUC": auc,
            "Probs": probs
        }

    def run_comparison(self):
        results = []
        all_single_prs_results = []

        sota_res = self.evaluate_single_prs(self.sota_col)
        results.append({"ID": f"Benchmark ({self.sota_label})", "AUC": sota_res["AUC"]})
        print(f"Benchmark {self.sota_label} AUC: {sota_res['AUC']:.4f}")

        best_single_res = None
        if self.run_best_single:
            best_single_auc = 0
            for prs in self.my_prs_cols:
                res = self.evaluate_single_prs(prs)
                all_single_prs_results.append({
                    "PRS_Name": res["Name"],
                    "AUC": res["AUC"],
                    "OR_per_SD": res["OR_per_SD"]
                })
                if res["AUC"] > best_single_auc:
                    best_single_auc = res["AUC"]
                    best_single_res = res
            results.append({"ID": f"Best Single ({best_single_res['Name']})", "AUC": best_single_auc})
        else:
            print("Skipping Best Single PRS evaluation (disabled)")
            best_single_res = {"Name": "Skipped", "AUC": 0.0, "Probs": np.zeros(len(self.data))}

        if self.run_elastic_net:
            enet_res = self.find_optimal_combination_elastic_net()
            results.append({"ID": "Elastic Net Fusion", "AUC": enet_res["AUC"]})
        else:
            print("Skipping Elastic Net (disabled)")
            enet_res = {"Name": "Elastic Net (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}

        if self.run_pca:
            pca_res = self.find_optimal_combination_pca()
            results.append({"ID": "PCLR (PCA Fusion)", "AUC": pca_res["AUC"]})
        else:
            print("Skipping PCA (disabled)")
            pca_res = {"Name": "PCA (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}

        if self.run_bayesian_pymc:
            bayes_res = self.find_optimal_combination_bayesian()
            if bayes_res["AUC"] > 0:
                results.append({"ID": "Bayesian Hierarchical (PyMC)", "AUC": bayes_res["AUC"]})
        else:
            print("Skipping Bayesian Hierarchical (PyMC) (disabled)")
            bayes_res = {"Name": "Bayesian PyMC (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}

        if self.run_bayesian_bambi:
            bambi_res = self.find_optimal_combination_bambi()
            if bambi_res["AUC"] > 0:
                results.append({"ID": "Bayesian Hierarchical (Bambi)", "AUC": bambi_res["AUC"]})
        else:
            print("Skipping Bayesian Hierarchical (Bambi) (disabled)")
            bambi_res = {"Name": "Bayesian Bambi (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}

        if self.run_with_sota:
            print("\n=== Running Models with SOTA PRS Included ===")
            
            if self.run_elastic_net:
                enet_sota_res = self.find_optimal_combination_elastic_net_with_sota()
                results.append({"ID": "Elastic Net + SOTA", "AUC": enet_sota_res["AUC"]})
            else:
                enet_sota_res = {"Name": "Elastic Net + SOTA (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}
            
            if self.run_pca:
                pca_sota_res = self.find_optimal_combination_pca_with_sota()
                results.append({"ID": "PCLR + SOTA", "AUC": pca_sota_res["AUC"]})
            else:
                pca_sota_res = {"Name": "PCA + SOTA (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}
            
            if self.run_bayesian_pymc:
                bayes_sota_res = self.find_optimal_combination_bayesian_with_sota()
                if bayes_sota_res["AUC"] > 0:
                    results.append({"ID": "Bayesian Hierarchical + SOTA (PyMC)", "AUC": bayes_sota_res["AUC"]})
            else:
                bayes_sota_res = {"Name": "Bayesian PyMC + SOTA (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}
            
            if self.run_bayesian_bambi:
                bambi_sota_res = self.find_optimal_combination_bambi_with_sota()
                if bambi_sota_res["AUC"] > 0:
                    results.append({"ID": "Bayesian Hierarchical + SOTA (Bambi)", "AUC": bambi_sota_res["AUC"]})
            else:
                bambi_sota_res = {"Name": "Bayesian Bambi + SOTA (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}
        else:
            print("\nSkipping Models with SOTA PRS (disabled)")
            enet_sota_res = {"Name": "Elastic Net + SOTA (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}
            pca_sota_res = {"Name": "PCA + SOTA (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}
            bayes_sota_res = {"Name": "Bayesian PyMC + SOTA (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}
            bambi_sota_res = {"Name": "Bayesian Bambi + SOTA (Skipped)", "AUC": 0.0, "Probs": np.zeros(len(self.data))}

        y_true = self.data[self.target_col]

        print("\nComparative statistics (vs SOTA):")
        comparison_list = []
        
        if self.run_best_single and best_single_res["AUC"] > 0:
            p_best = self._delong_test(best_single_res["Probs"], sota_res["Probs"], y_true)
            print(f"Best Single vs SOTA p-val: {p_best:.4e}")
            comparison_list.append({"Comparison": "Best Single vs SOTA", "p_value": p_best})
        
        if self.run_elastic_net and enet_res["AUC"] > 0:
            p_enet = self._delong_test(enet_res["Probs"], sota_res["Probs"], y_true)
            print(f"Elastic Net vs SOTA p-val: {p_enet:.4e}")
            comparison_list.append({"Comparison": "Elastic Net vs SOTA", "p_value": p_enet})
        
        if self.run_pca and pca_res["AUC"] > 0:
            p_pca = self._delong_test(pca_res["Probs"], sota_res["Probs"], y_true)
            print(f"PCLR vs SOTA p-val:        {p_pca:.4e}")
            comparison_list.append({"Comparison": "PCLR vs SOTA", "p_value": p_pca})

        if self.run_bayesian_pymc and bayes_res["AUC"] > 0:
            p_bayes = self._delong_test(bayes_res["Probs"], sota_res["Probs"], y_true)
            print(f"Bayesian (PyMC) vs SOTA p-val:   {p_bayes:.4e}")
            comparison_list.append({"Comparison": "Bayesian (PyMC) vs SOTA", "p_value": p_bayes})

        if self.run_bayesian_bambi and bambi_res["AUC"] > 0:
            p_bambi = self._delong_test(bambi_res["Probs"], sota_res["Probs"], y_true)
            print(f"Bayesian (Bambi) vs SOTA p-val:  {p_bambi:.4e}")
            comparison_list.append({"Comparison": "Bayesian (Bambi) vs SOTA", "p_value": p_bambi})

        if self.run_with_sota:
            print("\nWith SOTA comparisons (vs SOTA):")
            
            if self.run_elastic_net and enet_sota_res["AUC"] > 0:
                p_enet_sota = self._delong_test(enet_sota_res["Probs"], sota_res["Probs"], y_true)
                print(f"Elastic Net + SOTA vs SOTA p-val: {p_enet_sota:.4e}")
                comparison_list.append({"Comparison": "Elastic Net + SOTA vs SOTA", "p_value": p_enet_sota})
            
            if self.run_pca and pca_sota_res["AUC"] > 0:
                p_pca_sota = self._delong_test(pca_sota_res["Probs"], sota_res["Probs"], y_true)
                print(f"PCLR + SOTA vs SOTA p-val:        {p_pca_sota:.4e}")
                comparison_list.append({"Comparison": "PCLR + SOTA vs SOTA", "p_value": p_pca_sota})
            
            if self.run_bayesian_pymc and bayes_sota_res["AUC"] > 0:
                p_bayes_sota = self._delong_test(bayes_sota_res["Probs"], sota_res["Probs"], y_true)
                print(f"Bayesian (PyMC) + SOTA vs SOTA p-val:   {p_bayes_sota:.4e}")
                comparison_list.append({"Comparison": "Bayesian (PyMC) + SOTA vs SOTA", "p_value": p_bayes_sota})
            
            if self.run_bayesian_bambi and bambi_sota_res["AUC"] > 0:
                p_bambi_sota = self._delong_test(bambi_sota_res["Probs"], sota_res["Probs"], y_true)
                print(f"Bayesian (Bambi) + SOTA vs SOTA p-val:  {p_bambi_sota:.4e}")
                comparison_list.append({"Comparison": "Bayesian (Bambi) + SOTA vs SOTA", "p_value": p_bambi_sota})

        # Prepare statistical comparison dataframe
        stats_comparison = pd.DataFrame(comparison_list) if comparison_list else pd.DataFrame(columns=["Comparison", "p_value"])

        self._plot_roc(sota_res, best_single_res, enet_res, pca_res, bayes_res, bambi_res, enet_sota_res, pca_sota_res, bayes_sota_res, bambi_sota_res, y_true)
        
        summary_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
        
        # Save results if output directory is specified
        if self.output_dir:
            self._save_results(summary_df, pd.DataFrame(all_single_prs_results), stats_comparison,
                             sota_res, best_single_res, enet_res, pca_res, bayes_res, bambi_res,
                             enet_sota_res, pca_sota_res, bayes_sota_res, bambi_sota_res)
        
        return summary_df

    def _save_roc_tsv(self, plot_path, curves, y):
        """Persist (fpr, tpr, threshold, AUC) for every curve drawn in the
        accompanying figure into a sister TSV (same stem as the PDF)."""
        if not plot_path or not curves:
            return
        y_arr = np.asarray(y).astype(int)
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())
        rows = []
        for label, res in curves:
            if res is None or res.get("AUC", 0) == 0:
                continue
            probs = np.asarray(res["Probs"], dtype=float)
            fpr, tpr, thr = roc_curve(y_arr, probs)
            for f, t, th in zip(fpr, tpr, thr):
                rows.append({
                    "Model": label,
                    "Model_Internal_Name": res.get("Name", label),
                    "fpr": f, "tpr": t, "threshold": th,
                    "AUC": float(res["AUC"]),
                    "n_total": int(len(y_arr)),
                    "n_cases": n_pos, "n_controls": n_neg,
                    "SOTA_label": self.sota_label,
                    "Source_Plot": os.path.basename(plot_path),
                })
        if not rows:
            return
        tsv_path = os.path.splitext(plot_path)[0] + ".tsv"
        pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
        print(f"ROC source data saved to: {tsv_path}")

    def _plot_roc(self, sota, best, enet, pca, bayes, bambi, enet_sota, pca_sota, bayes_sota, bambi_sota, y):
        GREY_LIGHT = '#D3D3D3'
        
        # Helper function to plot a single curve
        def plot_curve(ax, res, label, style='-', color=None, drawn_curves=None):
            if res["AUC"] == 0:
                return False  # Skip if skipped
            fpr, tpr, _ = roc_curve(y, res["Probs"])
            ax.plot(fpr, tpr, label=f"{label} (AUC={res['AUC']:.3f})", linestyle=style, color=color, linewidth=2)
            if drawn_curves is not None:
                drawn_curves.append((label, res))
            return True
        
        def style_axes(ax):
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.spines["left"].set_color(GREY_LIGHT)
            ax.spines["left"].set_linewidth(2)
            ax.spines["bottom"].set_color(GREY_LIGHT)
            ax.spines["bottom"].set_linewidth(2)
            ax.tick_params(length=0)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_xlabel('False Positive Rate', size=18, weight='bold')
            ax.set_ylabel('True Positive Rate', size=18, weight='bold')
        
        drawn = []
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_curve(ax, sota, self.sota_label, color='#FF4D6FFF', drawn_curves=drawn)
        plot_curve(ax, best, f"Best Single: {best['Name']}", color='#86AD34FF', drawn_curves=drawn)
        plot_curve(ax, enet, "Multi PRS (Elastic Net)", color='#579EA4FF', drawn_curves=drawn)
        plot_curve(ax, pca, "Multi PRS (PCA Fusion)", color='#7E1A2FFF', drawn_curves=drawn)
        plot_curve(ax, bayes, "Multi PRS (Bayesian Hierarchical [PyMC])", color='#2D2651FF', drawn_curves=drawn)
        plot_curve(ax, bambi, "Multi PRS (Bayesian Hierarchical [Bambi]", color='#BD777AFF', drawn_curves=drawn)
        
        if self.run_with_sota:
            plot_curve(ax, enet_sota, "Multi PRS + SOTA (Elastic Net)", color='#579EA4FF', style='--', drawn_curves=drawn)
            plot_curve(ax, pca_sota, "Multi PRS + SOTA (PCA)", color='#7E1A2FFF', style='--', drawn_curves=drawn)
            plot_curve(ax, bayes_sota, "Multi PRS + SOTA (Bayesian [PyMC])", color='#2D2651FF', style='--', drawn_curves=drawn)
            plot_curve(ax, bambi_sota, "Multi PRS + SOTA (Bayesian [Bambi])", color='#BD777AFF', style='--', drawn_curves=drawn)

        ax.plot([0, 1], [0, 1], 'k:', alpha=0.5)
        style_axes(ax)
        ax.set_title(f'ROC Comparison: All Methods vs {self.sota_label} (N={len(y):,})', size=20, weight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=14, frameon=False)
        plt.tight_layout()
        
        if self.output_dir:
            plot_path = os.path.join(self.output_dir, "roc_comparison_all.pdf")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
            print(f"Combined ROC plot saved to: {plot_path}")
            self._save_roc_tsv(plot_path, drawn, y)
        
        plt.show()
        plt.close()
        
        # Individual comparisons: Each method vs SOTA
        methods_to_compare = [
            (best, f"Best Single: {best['Name']}", '#86AD34FF', "best_single"),
            (enet, "Multi PRS (Elastic Net)", '#579EA4FF', "elastic_net"),
            (pca, "Multi PRS (PCA Fusion)", '#7E1A2FFF', "pca"),
            (bayes, "Multi PRS (Bayesian Hierarchical [PyMC])", "#2D2651FF", "bayesian_pymc"),
            (bambi, "Multi PRS (Bayesian Hierarchical [Bambi])", "#BD777AFF", "bayesian_bambi")
        ]
        
        for method_res, method_label, method_color, filename_suffix in methods_to_compare:
            if method_res["AUC"] == 0:
                continue
            
            drawn = []
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot SOTA
            plot_curve(ax, sota, self.sota_label, color='#FF4D6FFF', style='--', drawn_curves=drawn)
            
            # Plot comparison method
            plot_curve(ax, method_res, method_label, color=method_color, style='-', drawn_curves=drawn)
            
            ax.plot([0, 1], [0, 1], 'k:', alpha=0.5)
            style_axes(ax)
            ax.set_title(f'ROC Comparison: {method_label} vs {self.sota_label} (N={len(y):,})', size=20, weight='bold', pad=20)
            ax.legend(loc="lower right", fontsize=14, frameon=False)
            plt.tight_layout()
            
            if self.output_dir:
                plot_path = os.path.join(self.output_dir, f"roc_comparison_{filename_suffix}_vs_sota.pdf")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
                print(f"ROC plot saved to: {plot_path}")
                self._save_roc_tsv(plot_path, drawn, y)
            
            plt.show()
            plt.close()
    
    def _save_results(self, summary_df, all_single_prs_df, stats_comparison_df,
                     sota_res, best_single_res, enet_res, pca_res, bayes_res, bambi_res,
                     enet_sota_res, pca_sota_res, bayes_sota_res, bambi_sota_res):
        print("\nSaving results")
        
        # Save summary results
        summary_path = os.path.join(self.output_dir, "auc_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"AUC summary saved to: {summary_path}")
        
        # Save all single PRS results (only if method was run)
        if self.run_best_single and not all_single_prs_df.empty:
            single_prs_path = os.path.join(self.output_dir, "all_single_prs_results.csv")
            all_single_prs_df.sort_values(by="AUC", ascending=False).to_csv(single_prs_path, index=False)
            print(f"All single PRS results saved to: {single_prs_path}")
        
        # Save statistical comparisons
        stats_path = os.path.join(self.output_dir, "statistical_comparisons.csv")
        stats_comparison_df.to_csv(stats_path, index=False)
        print(f"Statistical comparisons saved to: {stats_path}")
        
        # Save predicted probabilities for all methods (dynamically build based on what was run)
        probs_df = pd.DataFrame({
            "IID": self.data.index,
            f"SOTA_{sota_res['Name']}_Probs": sota_res["Probs"]
        })
        
        # Add each method's probabilities only if it was run
        if self.run_best_single and best_single_res["AUC"] > 0:
            probs_df[f"BestSingle_{best_single_res['Name']}_Probs"] = best_single_res["Probs"]
        
        if self.run_elastic_net and enet_res["AUC"] > 0:
            probs_df["ElasticNet_Probs"] = enet_res["Probs"]
        
        if self.run_pca and pca_res["AUC"] > 0:
            probs_df["PCLR_Probs"] = pca_res["Probs"]   
        
        if self.run_bayesian_pymc and bayes_res["AUC"] > 0:
            probs_df["Bayesian_PyMC_Probs"] = bayes_res["Probs"]
        
        if self.run_bayesian_bambi and bambi_res["AUC"] > 0:
            probs_df["Bayesian_Bambi_Probs"] = bambi_res["Probs"]
        
        # Add with_sota probabilities if enabled
        if self.run_with_sota:
            if self.run_elastic_net and enet_sota_res["AUC"] > 0:
                probs_df["ElasticNet_SOTA_Probs"] = enet_sota_res["Probs"]
            
            if self.run_pca and pca_sota_res["AUC"] > 0:
                probs_df["PCLR_SOTA_Probs"] = pca_sota_res["Probs"]
            
            if self.run_bayesian_pymc and bayes_sota_res["AUC"] > 0:
                probs_df["Bayesian_PyMC_SOTA_Probs"] = bayes_sota_res["Probs"]
            
            if self.run_bayesian_bambi and bambi_sota_res["AUC"] > 0:
                probs_df["Bayesian_Bambi_SOTA_Probs"] = bambi_sota_res["Probs"]
        
        # Also save the true target values
        probs_df[self.target_col] = self.data[self.target_col].values
        
        probs_path = os.path.join(self.output_dir, "predicted_probabilities.csv")
        probs_df.to_csv(probs_path, index=False)
        print(f"Predicted probabilities saved to: {probs_path}")
        
        # Save detailed results with AUC and OR for each method (dynamically build)
        detailed_results_list = []
        
        # Always include SOTA
        detailed_results_list.append({
            "Method": sota_res["Name"],
            "AUC": sota_res["AUC"],
            "OR_per_SD": sota_res.get("OR_per_SD", np.nan)
        })
        
        # Add each method only if it was run
        if self.run_best_single and best_single_res["AUC"] > 0:
            detailed_results_list.append({
                "Method": best_single_res["Name"],
                "AUC": best_single_res["AUC"],
                "OR_per_SD": best_single_res.get("OR_per_SD", np.nan)
            })
        
        if self.run_elastic_net and enet_res["AUC"] > 0:
            detailed_results_list.append({
                "Method": enet_res["Name"],
                "AUC": enet_res["AUC"],
                "OR_per_SD": np.nan
            })
        
        if self.run_pca and pca_res["AUC"] > 0:
            detailed_results_list.append({
                "Method": pca_res["Name"],
                "AUC": pca_res["AUC"],
                "OR_per_SD": np.nan
            })
        
        if self.run_bayesian_pymc and bayes_res["AUC"] > 0:
            detailed_results_list.append({
                "Method": bayes_res["Name"],
                "AUC": bayes_res["AUC"],
                "OR_per_SD": np.nan
            })
        
        if self.run_bayesian_bambi and bambi_res["AUC"] > 0:
            detailed_results_list.append({
                "Method": bambi_res["Name"],
                "AUC": bambi_res["AUC"],
                "OR_per_SD": np.nan
            })
        
        # Add with_sota methods if enabled
        if self.run_with_sota:
            if self.run_elastic_net and enet_sota_res["AUC"] > 0:
                detailed_results_list.append({
                    "Method": enet_sota_res["Name"],
                    "AUC": enet_sota_res["AUC"],
                    "OR_per_SD": np.nan
                })
            
            if self.run_pca and pca_sota_res["AUC"] > 0:
                detailed_results_list.append({
                    "Method": pca_sota_res["Name"],
                    "AUC": pca_sota_res["AUC"],
                    "OR_per_SD": np.nan
                })
            
            if self.run_bayesian_pymc and bayes_sota_res["AUC"] > 0:
                detailed_results_list.append({
                    "Method": bayes_sota_res["Name"],
                    "AUC": bayes_sota_res["AUC"],
                    "OR_per_SD": np.nan
                })
            
            if self.run_bayesian_bambi and bambi_sota_res["AUC"] > 0:
                detailed_results_list.append({
                    "Method": bambi_sota_res["Name"],
                    "AUC": bambi_sota_res["AUC"],
                    "OR_per_SD": np.nan
                })
        
        detailed_results = pd.DataFrame(detailed_results_list)
        detailed_path = os.path.join(self.output_dir, "detailed_method_results.csv")
        detailed_results.to_csv(detailed_path, index=False)
        print(f"Detailed method results saved to: {detailed_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PRS Comparison Pipeline - Compare multiple PRS strategies')
    
    # File paths
    parser.add_argument('--pth_covars', type=str, default="/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/covars/nonDisc_caucasian_king0p0625_V0.tsv",
                        help='Path to covariates TSV file')
    parser.add_argument('--prs_res_root', type=str, default="/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc",
                        help='Root directory containing PRS results')
    parser.add_argument('--pth_prs_processed', type=str, default="/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/all_latents_PRS.tsv",
                        help='Path to pre-processed PRS table in TSV format. If provided, bypasses individual file processing')
    parser.add_argument('--rds_pres_prefix', type=str, default='run_ext_basic_king0p0625_lw_gw_indep_FiltMAF_',
                        help='Prefix for RDS PRS files')
    parser.add_argument('--rds_pres_suffix', type=str, default='.fullDS.auto.mod.LDPred2.rds',
                        help='Suffix for RDS PRS files (e.g., .fullDS.auto.mod.LDPred2.rds for auto model)')
    parser.add_argument('--rds_tag_prs', type=str, default='auto.mod',
                        help='Tag for PRS model type (e.g., auto.mod, inf.mod, grid.mod)')
    parser.add_argument('--tag_data', type=str, default='resNdata.basic',
                        help='Tag for data file')
    parser.add_argument('--tag_prs', type=str, default='pred_auto',
                        help='Tag for PRS prediction')
    parser.add_argument('--pth_prs_sota', type=str, default="/project/ukbblatent/clinicaldata/v1.1.0_seventh_basket/PRS_82779_MD_12_11_2025_12_17_48.tsv",
                        help='Path to SOTA PRS TSV file')
    parser.add_argument('--prs_sota_orig_cols', type=str, nargs='+', default=['f.26223.0.0', 'f.26227.0.0', 'f.26212.0.0', 'f.26244.0.0', 'f.26285.0.0'],
                        help='Original column names for SOTA PRS (CVD and CAD)')
    parser.add_argument('--prs_sota_col_tags', type=str, nargs='+', default=['prs_sota_CVD', 'prs_sota_CAD', 'prs_sota_AFib', 'prs_sota_HT', 'prs_sota_T2D'],
                        help='Renamed column tags for SOTA PRS')
    parser.add_argument('--pth_dis', type=str, default="/project/ukbblatent/clinicaldata/binary_disease_cohorts/F20208v3_nonDiscov/caucasian_king0p0625_grouped/newcovsets/V0v2/atherosclerotic.csv",
                        help='Path to disease cohort CSV file')
    parser.add_argument('--pth_out_root', type=str, default="/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/comparisons",
                        help='Output root directory for results')
    parser.add_argument('--pth_out_dir', type=str, default="ukb_26223_CVD_V0v2",
                        help='Output directory for results, param names will be concatenated to this')
    
    # PRSEvaluator parameters
    parser.add_argument('--target_col', type=str, default='BinCAT_Disease',
                        help='Name of the binary disease outcome column')
    parser.add_argument('--covariates', type=str, nargs='+', default=['Age', 'BMI'],
                        help='List of continuous covariate column names')
    parser.add_argument('--cat_covariates', type=str, nargs='+', default=['Sex', 'CAT_Smoking'],
                        help='List of categorical covariate column names')
    parser.add_argument('--sota_col', type=str, default='prs_sota_CVD',
                        help='Name of the SOTA PRS column to use as benchmark')
    parser.add_argument('--sota_label', type=str, default='SOTA Benchmark',
                        help='Label to display for SOTA benchmark in plots')
    parser.add_argument('--orthogonalise', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to orthogonalise PRS against covariates (use --no-orthogonalise to disable)')
    parser.add_argument('--use_variational_inference', action=argparse.BooleanOptionalAction, default=False,
                        help='Use Variational Inference (ADVI) instead of MCMC for Bayesian methods. ~10-50× faster but approximate (use --use-variational-inference to enable)')
    
    # Method selection flags
    parser.add_argument('--run_best_single', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to evaluate best single PRS (use --no-run_best_single to disable)')
    parser.add_argument('--run_elastic_net', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to run Elastic Net method (use --no-run_elastic_net to disable)')
    parser.add_argument('--run_pca', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to run PCA method (use --no-run_pca to disable)')
    parser.add_argument('--run_bayesian_pymc', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to run Bayesian Hierarchical (PyMC) method (use --no-run_bayesian_pymc to disable)')
    parser.add_argument('--run_bayesian_bambi', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to run Bayesian Hierarchical (Bambi) method (use --no-run_bayesian_bambi to disable)')
    parser.add_argument('--run_with_sota', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to run additional models combining multi PRS with SOTA PRS (use --run-with-sota to enable)')
    
    args = parser.parse_args()
    
    pth_covars = args.pth_covars
    prs_res_root = args.prs_res_root
    pth_prs_processed = args.pth_prs_processed
    rds_pres_prefix = args.rds_pres_prefix
    rds_pres_suffix = args.rds_pres_suffix
    rds_tag_prs = args.rds_tag_prs
    tag_data = args.tag_data
    tag_prs = args.tag_prs
    pth_prs_sota = args.pth_prs_sota
    prs_sota_orig_cols = args.prs_sota_orig_cols
    prs_sota_col_tags = args.prs_sota_col_tags
    pth_dis = args.pth_dis
    pth_out_root = args.pth_out_root
    pth_out_dir = args.pth_out_dir
    
    target_col = args.target_col
    covariates = args.covariates
    cat_covariates = args.cat_covariates
    sota_col = args.sota_col
    sota_label = args.sota_label
    orthogonalise = args.orthogonalise
    use_variational_inference = args.use_variational_inference
    
    run_best_single = args.run_best_single
    run_elastic_net = args.run_elastic_net
    run_pca = args.run_pca
    run_bayesian_pymc = args.run_bayesian_pymc
    run_bayesian_bambi = args.run_bayesian_bambi
    run_with_sota = args.run_with_sota

    pth_out_dir = f"{pth_out_dir}_ortho" if orthogonalise else f"{pth_out_dir}_noortho"
    pth_out_dir += "_vi" if use_variational_inference else "_mcmc"

    covar = pd.read_table(pth_covars, sep="\t")
    covar.set_index("IID", inplace=True)

    if pth_prs_processed:
        prs = pd.read_table(pth_prs_processed, sep="\t")
        prs.set_index("IID", inplace=True)
        print(f"Loaded {len(prs)} samples with {len(prs.columns)} PRS features")
    else:
        print("Processing individual PRS files...")
        pths_prs = glob(f"{prs_res_root}/{rds_pres_prefix}*{rds_pres_suffix}")

        prs_store = []
        for pth_prs in tqdm(pths_prs):
            latent = pth_prs.split(rds_pres_prefix)[1].split(rds_pres_suffix)[0]
            
            data = pyreadr.read_r(pth_prs.replace(rds_tag_prs, tag_data))[None]
            prs_latent = data[['IID']].copy()
            r_obj = read_rds(pth_prs)
            #prs["PRS"] = r_obj['data'][r_obj['attributes']['names']['data'].index(tag_prs)]['data']
            prs_latent[f"PRS:{latent}"] = r_obj[tag_prs].copy()
            prs_latent.set_index("IID", inplace=True)
            prs_store.append(prs_latent)
        prs = pd.concat(prs_store, axis=1)
        print(f"Processed {len(prs_store)} PRS files")

    prs_sota = pd.read_table(pth_prs_sota, sep="\t")
    prs_sota.set_index("f.eid", inplace=True)
    prs_sota.rename(columns=dict(zip(prs_sota_orig_cols, prs_sota_col_tags)), inplace=True)
    prs_sota.drop(columns=[col for col in prs_sota.columns if col not in prs_sota_col_tags], inplace=True)

    disDF = pd.read_csv(pth_dis)
    disDF.set_index("eid", inplace=True)
    disDF = disDF[[target_col]]

    full_output_dir = os.path.join(pth_out_root, pth_out_dir)
    
    evaluator = PRSEvaluator(
        covar, prs, prs_sota, disDF,
        target_col=target_col,
        covariates=covariates,
        cat_covariates=cat_covariates,
        sota_col=sota_col,
        sota_label=sota_label,
        orthogonalise=orthogonalise,
        output_dir=full_output_dir,
        use_variational_inference=use_variational_inference,
        run_best_single=run_best_single,
        run_elastic_net=run_elastic_net,
        run_pca=run_pca,
        run_bayesian_pymc=run_bayesian_pymc,
        run_bayesian_bambi=run_bayesian_bambi,
        run_with_sota=run_with_sota
    )
    summary = evaluator.run_comparison()
    print("\n", summary)