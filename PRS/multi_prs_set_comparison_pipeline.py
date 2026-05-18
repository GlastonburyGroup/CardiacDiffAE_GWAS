# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
import os
import argparse
from itertools import combinations

import pyreadr
from rds2py import read_rds

import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
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


# Custom virtual subgroups: select specific features from across all PRS sets.
VIRTUAL_SUBGROUPS = {
    "Long axis 4ch IDPs": [
        "PRS:LV_longitudinal_strain_global",
        "PRS:RA_maximum_volume",
        "PRS:RA_ejection_fraction" #also, RA min vol, stroke vol
    ],
    "Long axis 4ch+2ch IDPs": [
        "PRS:LA_minimum_volume",
        "PRS:LA_ejection_fraction" #also, LA min vol, stroke vol
    ],
    "Long axis IDPs": [
        "PRS:LV_longitudinal_strain_global",
        "PRS:RA_maximum_volume",
        "PRS:RA_ejection_fraction", 
        "PRS:LA_minimum_volume",
        "PRS:LA_ejection_fraction" #also, RA min vol, stroke vol, LA min vol, stroke vol
    ],
    "Short axis IDPs": [
        "PRS:LV_circumferential_strain_global",
        "PRS:LV_end_systolic_volume",
        "PRS:LV_myocardial_mass",
        "PRS:LV_radial_strain_global",
        "PRS:LV_ejection_fraction",
        "PRS:RV_end_systolic_volume",
        "PRS:RV_ejection_fraction",
        "PRS:RV_end_diastolic_volume",
        "PRS:LV_end_diastolic_volume",
        "PRS:LV_mean_myocardial_wall_thickness_global",
        "PRS:RV_stroke_volume" #also, LV stroke vol e LV cardiac output
    ]
}


class MultiPRSSetEvaluator:
    """
    A comprehensive engine for comparing multiple PRS sets against each other.
    Each PRS set contains multiple PRS values and is processed identically,
    enabling direct performance comparison between different PRS generation approaches.
    
    Handles data harmonisation, logistic modelling, Elastic Net fusion,
    Principal Component Logistic Regression (PCLR), Bayesian Hierarchical modelling,
    and statistical significance testing across multiple PRS sets.
    """

    def __init__(self, covar_df: pd.DataFrame, prs_sets_dict: Dict[str, pd.DataFrame],
                 disease_df: pd.DataFrame,
                 target_col: str = "BinCAT_Disease",
                 covariates: List[str] = None,
                 cat_covariates: List[str] = None,
                 reference_set: str = None,
                 virtual_subgroups: Dict[str, List[str]] = None,
                 ignore_features: List[str] = None,
                 orthogonalise: bool = False,
                 use_covariates_as_inputs: bool = True,
                 output_dir: str = None,
                 use_variational_inference: bool = False,
                 run_best_single: bool = True,
                 run_elastic_net: bool = True,
                 run_pca: bool = True,
                 run_bayesian_pymc: bool = True,
                 run_bayesian_bambi: bool = True):
        """
        Initialises the multi-PRS set evaluator.

        :param covar_df: DataFrame with covariates (indexed by IID)
        :param prs_sets_dict: Dictionary mapping set names to PRS DataFrames
                              e.g., {"My_Custom": df1, "LDPred2": df2, "PRS_CS": df3}
        :param disease_df: DataFrame with disease outcomes (indexed by IID)
        :param target_col: Name of the binary disease outcome column
        :param covariates: List of continuous covariate column names
        :param cat_covariates: List of categorical covariate column names
        :param reference_set: Name of the reference PRS set for statistical comparisons
                             If None, will use the first set in prs_sets_dict
        :param virtual_subgroups: Dictionary mapping subgroup names to lists of feature names
                                 e.g., {"Cardiac_Core": ["RV_end_diastolic_volume", "LV_circumferential_strain_global"]}
                                 Features can come from any of the loaded PRS sets
        :param ignore_features: List of feature names (original, unprefixed) to exclude from all PRS sets
                               If a feature name appears in multiple sets, it will be removed from all of them
        :param orthogonalise: If True, regresses all PRS against Covariates and uses RESIDUALS
                             This removes covariate effects from PRS before modelling
        :param use_covariates_as_inputs: If True, includes covariates as input variables in models
                                         If False, models use only PRS features as inputs
                                         Works independently from orthogonalise flag
        :param output_dir: Directory to save all output files (plots, dataframes, etc.)
        :param use_variational_inference: If True, uses ADVI instead of MCMC for Bayesian methods
        :param run_best_single: If True, evaluates best single PRS for each set
        :param run_elastic_net: If True, runs Elastic Net method for each set
        :param run_pca: If True, runs PCA method for each set
        :param run_bayesian_pymc: If True, runs Bayesian Hierarchical (PyMC) method for each set
        :param run_bayesian_bambi: If True, runs Bayesian Hierarchical (Bambi) method for each set
        """
        print("Initialising Multi-PRS Set Evaluator")
        
        prefixed_prs_sets = {}
        for set_name, prs_df in prs_sets_dict.items():
            prefixed_df = prs_df.copy()
            prefixed_df.columns = [f"{set_name}::{col}" for col in prefixed_df.columns]
            prefixed_prs_sets[set_name] = prefixed_df
            print(f"Prefixed {len(prefixed_df.columns)} columns for set '{set_name}'")
        
        # Filter out ignored features if specified
        if ignore_features:
            for set_name, prefixed_df in prefixed_prs_sets.items():
                # Find columns that match the ignore list (check unprefixed names)
                cols_to_drop = []
                for col in prefixed_df.columns:
                    original_name = col.split("::", 1)[1] if "::" in col else col
                    if original_name in ignore_features:
                        cols_to_drop.append(col)
                
                if cols_to_drop:
                    prefixed_prs_sets[set_name] = prefixed_df.drop(columns=cols_to_drop)
        
        # Merge all PRS sets with covariates and disease data
        self.raw_data = covar_df.join(list(prefixed_prs_sets.values()) + [disease_df], how='inner')

        initial_n = len(self.raw_data)
        self.data = self.raw_data.dropna()
        print(f"Data merged. N={len(self.data)} (Dropped {initial_n - len(self.data)} rows with missing data).")

        self.target_col = target_col
        self.covariates = covariates if covariates is not None else ["Age", "Sex", "BMI"]
        self.cat_covariates = cat_covariates if cat_covariates is not None else ["CAT_Smoking"]
        
        self.prs_sets = {name: list(df.columns) for name, df in prefixed_prs_sets.items()}
        self.prs_set_names = list(self.prs_sets.keys())
        self.use_covariates_as_inputs = use_covariates_as_inputs
        
        self.reference_set = reference_set if reference_set is not None else self.prs_set_names[0]
        if self.reference_set not in self.prs_set_names:
            raise ValueError(f"Reference set '{self.reference_set}' not found in PRS sets: {self.prs_set_names}")
        
        print(f"PRS Sets loaded: {self.prs_set_names}")
        print(f"Reference set for comparisons: {self.reference_set}")
        for name, cols in self.prs_sets.items():
            print(f"  - {name}: {len(cols)} PRS features")
        
        # Create virtual subgroups if specified
        if virtual_subgroups:
            self._create_virtual_subgroups(virtual_subgroups)
        
        self.use_variational_inference = use_variational_inference
        
        self.run_best_single = run_best_single
        self.run_elastic_net = run_elastic_net
        self.run_pca = run_pca
        self.run_bayesian_pymc = run_bayesian_pymc
        self.run_bayesian_bambi = run_bayesian_bambi
        
        self.output_dir = output_dir
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {self.output_dir}")

        self.data = pd.get_dummies(self.data, columns=self.cat_covariates, drop_first=True)
        new_covars = [c for c in self.data.columns if any(cat in c for cat in self.cat_covariates)]
        self.final_covariates = self.covariates + new_covars

        if orthogonalise:
            self._orthogonalise_prs_features()

        self._preprocess_data()

    def _create_virtual_subgroups(self, virtual_subgroups: Dict[str, List[str]]):
        """Add virtual PRS subgroups (selections across loaded sets) to the comparison."""
        print("Creating virtual subgroups")
        
        # Get all available features from merged data
        all_available_features = set(self.data.columns)
        
        for subgroup_name, feature_list in virtual_subgroups.items():
            # For each requested feature, find it in the prefixed columns
            # Match pattern: "SetName::OriginalFeatureName"
            available_features = []
            missing_features = []
            
            for orig_feature in feature_list:
                # Search for this feature across all prefixed columns
                matched = False
                for col in all_available_features:
                    # Check if column matches pattern "SomePRSSet::orig_feature"
                    if "::" in col and col.split("::", 1)[1] == orig_feature:
                        available_features.append(col)
                        matched = True
                        break
                
                if not matched:
                    missing_features.append(orig_feature)
            
            if missing_features:
                print(f"  Warning: Subgroup '{subgroup_name}' - {len(missing_features)} features not found: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            
            if not available_features:
                print(f"  Error: Subgroup '{subgroup_name}' has no valid features. Skipping.")
                continue
            
            # Check if subgroup name conflicts with existing set names
            if subgroup_name in self.prs_set_names:
                print(f"  Error: Subgroup name '{subgroup_name}' conflicts with existing PRS set. Skipping.")
                continue
            
            # Add subgroup to prs_sets
            self.prs_sets[subgroup_name] = available_features
            self.prs_set_names.append(subgroup_name)
            
            print(f"  ✓ Created subgroup '{subgroup_name}': {len(available_features)}/{len(feature_list)} features")
            print(f"    Features: {available_features[:5]}{'...' if len(available_features) > 5 else ''}")
        
        print(f"\nTotal PRS sets (including subgroups): {len(self.prs_set_names)}")

    def _orthogonalise_prs_features(self):
        print("Performing feature orthogonalisation (residualisation)")
        X_nuisance = self.data[self.final_covariates].astype(float)

        all_prs_cols = []
        for prs_cols in self.prs_sets.values():
            all_prs_cols.extend(prs_cols)

        lr = LinearRegression()
        for col in all_prs_cols:
            lr.fit(X_nuisance, self.data[col])
            self.data[col] = self.data[col] - lr.predict(X_nuisance)

        print(f"PRS features from all {len(self.prs_sets)} sets residualised against {len(self.final_covariates)} covariates.")

    def _preprocess_data(self):
        """
        Standardises continuous variables only (Age, BMI, PRS scores).
        One-hot encoded categorical variables should NOT be standardised.
        """
        scaler = StandardScaler()
        continuous_covars = [c for c in self.covariates if c in self.data.columns]
        
        # Collect all PRS columns from all sets
        all_prs_cols = []
        for prs_cols in self.prs_sets.values():
            all_prs_cols.extend(prs_cols)
        
        cols_to_scale = continuous_covars + all_prs_cols
        self.data[cols_to_scale] = scaler.fit_transform(self.data[cols_to_scale])
        print("Standardisation complete: Z-scores calculated for continuous variables.")

    def _get_original_feature_name(self, prefixed_name: str) -> str:
        """
        Extracts the original feature name from a prefixed column name.
        E.g., "My_Custom::PRS:LV_strain" -> "PRS:LV_strain"
        """
        if "::" in prefixed_name:
            return prefixed_name.split("::", 1)[1]
        return prefixed_name
    
    def _calculate_calibration_metrics(self, probs, y_true) -> Dict:
        """
        Calculates calibration metrics: Brier Score and Log Loss.
        
        :param probs: Predicted probabilities
        :param y_true: True labels
        :return: Dictionary with Brier Score and Log Loss
        """
        try:
            brier = brier_score_loss(y_true, probs)
            logloss = log_loss(y_true, probs)
            return {"Brier_Score": brier, "Log_Loss": logloss}
        except (ValueError, RuntimeError) as e:
            print(f"Warning: Calibration metric calculation failed: {e}")
            return {"Brier_Score": np.nan, "Log_Loss": np.nan}
    
    def _delong_test(self, preds_a, preds_b, target):
        """
        Calculates DeLong's test p-value for correlated ROC curves.
        """
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
        """Evaluates a single PRS using logistic regression (with or without covariates)."""
        # Construct feature list based on use_covariates_as_inputs flag
        if self.use_covariates_as_inputs:
            feature_cols = self.final_covariates + [prs_col_name]
        else:
            feature_cols = [prs_col_name]
        
        X = self.data[feature_cols].astype(float)
        y = self.data[self.target_col]

        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]

        return {
            "Name": prs_col_name,
            "OriginalName": self._get_original_feature_name(prs_col_name),
            "AUC": roc_auc_score(y, probs),
            "Probs": probs,
            "OR_per_SD": np.exp(model.coef_[0][-1])
        }

    def find_best_single_prs(self, prs_cols: List[str], set_name: str) -> Dict:
        """Finds the best performing single PRS from a given set."""
        print(f"\nBest single PRS: {set_name}")
        best_single_auc = 0
        best_single_res = None
        all_results = []
        
        for prs_col in prs_cols:
            res = self.evaluate_single_prs(prs_col)
            all_results.append(res)
            if res["AUC"] > best_single_auc:
                best_single_auc = res["AUC"]
                best_single_res = res
        
        original_name = best_single_res['OriginalName']
        print(f"Best single PRS: {original_name} (AUC={best_single_auc:.4f})")
        
        return {
            "Name": f"Best Single ({set_name}: {original_name})",
            "AUC": best_single_auc,
            "Probs": best_single_res["Probs"],
            "SetName": set_name,
            "AllResults": all_results
        }

    def find_optimal_combination_elastic_net(self, prs_cols: List[str], set_name: str) -> Dict:
        print(f"\nElastic Net: {set_name}")
        if self.use_covariates_as_inputs:
            feature_cols = self.final_covariates + prs_cols
            n_covar_features = len(self.final_covariates)
        else:
            feature_cols = prs_cols
            n_covar_features = 0
        
        X = self.data[feature_cols].astype(float)
        y = self.data[self.target_col]

        el_net = LogisticRegressionCV(
            cv=5, penalty='elasticnet', solver='saga',
            l1_ratios=[0.1, 0.5, 0.7, 0.95], max_iter=5000, n_jobs=-1, random_state=42
        )
        el_net.fit(X, y)
        probs = el_net.predict_proba(X)[:, 1]

        coefs = el_net.coef_[0][n_covar_features:]
        n_selected = np.sum(coefs != 0)
        print(f"Elastic Net ({set_name}) retained {n_selected}/{len(prs_cols)} PRS features.")

        return {
            "Name": f"Elastic Net ({set_name})",
            "AUC": roc_auc_score(y, probs),
            "Probs": probs,
            "SetName": set_name,
            "N_Selected": n_selected
        }

    def find_optimal_combination_pca(self, prs_cols: List[str], set_name: str, variance_threshold: float = 0.95) -> Dict:
        print(f"\nPCA: {set_name}")

        X_prs = self.data[prs_cols].astype(float)
        pca = PCA(n_components=variance_threshold)
        X_pca = pca.fit_transform(X_prs)

        n_components = X_pca.shape[1]
        print(f"PCA ({set_name}) reduced {len(prs_cols)} PRS traits to {n_components} orthogonal components.")

        pca_cols = [f"PC_{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=self.data.index)

        # Construct feature list based on use_covariates_as_inputs flag
        if self.use_covariates_as_inputs:
            X_final = pd.concat([self.data[self.final_covariates].astype(float), df_pca], axis=1)
        else:
            X_final = df_pca
        
        y = self.data[self.target_col]

        model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
        model.fit(X_final, y)
        probs = model.predict_proba(X_final)[:, 1]

        return {
            "Name": f"PCA Fusion ({set_name})",
            "AUC": roc_auc_score(y, probs),
            "Probs": probs,
            "SetName": set_name,
            "N_Components": n_components
        }

    def find_optimal_combination_bayesian(self, prs_cols: List[str], set_name: str) -> Dict:
        if not BAYES_AVAILABLE:
            return {"Name": f"Bayesian PyMC ({set_name}) - Skipped", "AUC": 0.0, "Probs": np.zeros(len(self.data)), "SetName": set_name}

        print(f"\nBayesian Hierarchical (PyMC): {set_name}")

        try:
            X_cov = self.data[self.final_covariates].astype(float).values
            X_prs = self.data[prs_cols].astype(float).values
            y_obs = self.data[self.target_col].values
        except ValueError as e:
            print(f"[ERROR] Data conversion failed for {set_name}: {e}")
            return {"Name": f"Bayesian PyMC ({set_name}) - Failed", "AUC": 0.0, "Probs": np.zeros(len(self.data)), "SetName": set_name}

        n_cov = X_cov.shape[1]
        n_prs = X_prs.shape[1]

        with pm.Model() as hierarchical_model:
            alpha = pm.Normal("alpha", mu=0, sigma=2)
            
            if self.use_covariates_as_inputs and n_cov > 0:
                beta_cov = pm.Normal("beta_cov", mu=0, sigma=2, shape=n_cov)
                cov_effect = pm.math.dot(X_cov, beta_cov)
            else:
                cov_effect = 0
            
            tau = pm.HalfNormal("tau", sigma=1)
            beta_prs = pm.Normal("beta_prs", mu=0, sigma=tau, shape=n_prs)
            prs_effect = pm.math.dot(X_prs, beta_prs)
            
            mu = alpha + cov_effect + prs_effect
            y_est = pm.Bernoulli("y_est", logit_p=mu, observed=y_obs)

            if self.use_variational_inference:
                print(f"Using Variational Inference (ADVI) for {set_name}...")
                inference = pm.ADVI()
                approx = pm.fit(n=20000, method=inference, progressbar=True)
                trace = approx.sample(2000)
                
                posterior_alpha = float(trace.posterior["alpha"].mean().values)
                if self.use_covariates_as_inputs and n_cov > 0:
                    posterior_cov = trace.posterior["beta_cov"].mean(dim="draw").values.flatten()
                else:
                    posterior_cov = np.zeros(n_cov)
                posterior_prs = trace.posterior["beta_prs"].mean(dim="draw").values.flatten()
            else:
                print(f"Using MCMC (NUTS) for {set_name}...")
                trace = pm.sample(draws=1000, tune=1000, target_accept=0.9, chains=4, cores=4, progressbar=True)
                
                posterior_alpha = float(trace.posterior["alpha"].mean(dim=["chain", "draw"]).values)
                if self.use_covariates_as_inputs and n_cov > 0:
                    posterior_cov = trace.posterior["beta_cov"].mean(dim=["chain", "draw"]).values
                else:
                    posterior_cov = np.zeros(n_cov)
                posterior_prs = trace.posterior["beta_prs"].mean(dim=["chain", "draw"]).values

            if self.use_covariates_as_inputs and n_cov > 0:
                logit_p = posterior_alpha + np.dot(X_cov, posterior_cov) + np.dot(X_prs, posterior_prs)
            else:
                logit_p = posterior_alpha + np.dot(X_prs, posterior_prs)
            probs = 1 / (1 + np.exp(-logit_p))

        inference_method = "VI-ADVI" if self.use_variational_inference else "MCMC-NUTS"
        print(f"Bayesian PyMC ({set_name}) Sampling Complete ({inference_method}).")
        auc = roc_auc_score(y_obs, probs)

        return {
            "Name": f"Bayesian PyMC ({set_name})",
            "AUC": auc,
            "Probs": probs,
            "SetName": set_name
        }

    def find_optimal_combination_bambi(self, prs_cols: List[str], set_name: str) -> Dict:
        if not BAMBI_AVAILABLE:
            return {"Name": f"Bayesian Bambi ({set_name}) - Skipped", "AUC": 0.0, "Probs": np.zeros(len(self.data)), "SetName": set_name}

        print(f"\nBayesian Hierarchical (Bambi): {set_name}")

        prs_formula_part = " + ".join([f"`{col}`" for col in prs_cols])
        
        if self.use_covariates_as_inputs:
            cov_formula_part = " + ".join([f"`{col}`" for col in self.final_covariates])
            formula = f"`{self.target_col}` ~ {cov_formula_part} + {prs_formula_part}"
        else:
            formula = f"`{self.target_col}` ~ {prs_formula_part}"
        
        nested_sigma = bmb.Prior("HalfNormal", sigma=1)
        hierarchical_prior = bmb.Prior("Normal", mu=0, sigma=nested_sigma)
        my_priors = {f"`{prs_col}`": hierarchical_prior for prs_col in prs_cols}

        model = bmb.Model(formula, data=self.data, family="bernoulli", priors=my_priors)
        
        if self.use_variational_inference:
            print(f"Using Variational Inference (ADVI) for {set_name}...")
            approx = model.fit(method="vi", inference_method="advi", n=20000, random_seed=42, progressbar=True)
            inference_method = "VI-ADVI"
            
            trace = approx.sample(2000)
            model.predict(trace, kind="response", inplace=True)
            probs = trace.posterior_predictive[self.target_col].mean(dim="draw").values.flatten()
        else:
            print(f"Using MCMC (NUTS) for {set_name}...")
            results = model.fit(draws=1000, tune=1000, target_accept=0.9, chains=4, cores=4, random_seed=42, progressbar=True)
            inference_method = "MCMC-NUTS"
            
            model.predict(results, kind="response")
            probs = results.posterior[f"{self.target_col}_mean"].mean(dim=("chain", "draw")).values

        print(f"Bayesian Bambi ({set_name}) Sampling Complete ({inference_method}).")
        auc = roc_auc_score(self.data[self.target_col], probs)
        
        return {
            "Name": f"Bayesian Bambi ({set_name})",
            "AUC": auc,
            "Probs": probs,
            "SetName": set_name
        }

    def run_comparison(self):
        method_results = {
            "best_single": {},
            "elastic_net": {},
            "pca": {},
            "bayesian_pymc": {},
            "bayesian_bambi": {}
        }
        
        all_single_prs_results = []
        
        for set_name in self.prs_set_names:
            prs_cols = self.prs_sets[set_name]
            print(f"\nProcessing PRS Set: {set_name} ({len(prs_cols)} features)")
            
            if self.run_best_single:
                result = self.find_best_single_prs(prs_cols, set_name)
                method_results["best_single"][set_name] = result
                
                for res in result["AllResults"]:
                    all_single_prs_results.append({
                        "Set": set_name,
                        "PRS_Name": res["OriginalName"],
                        "PRS_Full_Name": res["Name"],
                        "AUC": res["AUC"],
                        "OR_per_SD": res["OR_per_SD"]
                    })
            else:
                method_results["best_single"][set_name] = {
                    "Name": f"Best Single ({set_name}) - Skipped",
                    "AUC": 0.0,
                    "Probs": np.zeros(len(self.data)),
                    "SetName": set_name
                }
            
            if self.run_elastic_net:
                result = self.find_optimal_combination_elastic_net(prs_cols, set_name)
                method_results["elastic_net"][set_name] = result
            else:
                method_results["elastic_net"][set_name] = {
                    "Name": f"Elastic Net ({set_name}) - Skipped",
                    "AUC": 0.0,
                    "Probs": np.zeros(len(self.data)),
                    "SetName": set_name
                }
            
            if self.run_pca:
                result = self.find_optimal_combination_pca(prs_cols, set_name)
                method_results["pca"][set_name] = result
            else:
                method_results["pca"][set_name] = {
                    "Name": f"PCA ({set_name}) - Skipped",
                    "AUC": 0.0,
                    "Probs": np.zeros(len(self.data)),
                    "SetName": set_name
                }
            
            if self.run_bayesian_pymc:
                result = self.find_optimal_combination_bayesian(prs_cols, set_name)
                method_results["bayesian_pymc"][set_name] = result
            else:
                method_results["bayesian_pymc"][set_name] = {
                    "Name": f"Bayesian PyMC ({set_name}) - Skipped",
                    "AUC": 0.0,
                    "Probs": np.zeros(len(self.data)),
                    "SetName": set_name
                }
            
            if self.run_bayesian_bambi:
                result = self.find_optimal_combination_bambi(prs_cols, set_name)
                method_results["bayesian_bambi"][set_name] = result
            else:
                method_results["bayesian_bambi"][set_name] = {
                    "Name": f"Bayesian Bambi ({set_name}) - Skipped",
                    "AUC": 0.0,
                    "Probs": np.zeros(len(self.data)),
                    "SetName": set_name
                }
        
        y_true = self.data[self.target_col]
        comparison_list = []
        
        for method_name, method_data in method_results.items():
            if not any(res["AUC"] > 0 for res in method_data.values()):
                continue
                
            print(f"\n{method_name.replace('_', ' ').title()} comparisons:")
            
            for set_a, set_b in combinations(self.prs_set_names, 2):
                res_a = method_data[set_a]
                res_b = method_data[set_b]
                
                if res_a["AUC"] > 0 and res_b["AUC"] > 0:
                    p_val = self._delong_test(res_a["Probs"], res_b["Probs"], y_true)
                    print(f"{set_a} vs {set_b}: AUC_diff={res_a['AUC']-res_b['AUC']:+.4f}, p={p_val:.4e}")
                    comparison_list.append({
                        "Method": method_name,
                        "Set_A": set_a,
                        "Set_B": set_b,
                        "AUC_A": res_a["AUC"],
                        "AUC_B": res_b["AUC"],
                        "AUC_Diff": res_a["AUC"] - res_b["AUC"],
                        "p_value": p_val
                    })
        
        stats_comparison = pd.DataFrame(comparison_list) if comparison_list else pd.DataFrame()
        
        for method_name, method_data in method_results.items():
            if not any(res["AUC"] > 0 for res in method_data.values()):
                continue
                
            print(f"\n{method_name.replace('_', ' ').title()}:")
            for set_name, result in method_data.items():
                if result["AUC"] > 0:
                    calibration_metrics = self._calculate_calibration_metrics(result["Probs"], y_true)
                    result["Brier_Score"] = calibration_metrics["Brier_Score"]
                    result["Log_Loss"] = calibration_metrics["Log_Loss"]
                    print(f"{set_name}: Brier={calibration_metrics['Brier_Score']:.4f}, LogLoss={calibration_metrics['Log_Loss']:.4f}")
        
        # Generate plots
        self._plot_roc_comparisons(method_results, y_true)
        self._plot_calibration_curves(method_results, y_true)
        self._plot_decision_curve_analysis(method_results, y_true)
        
        # Create summary DataFrame
        summary_list = []
        for method_name, method_data in method_results.items():
            for set_name, result in method_data.items():
                if result["AUC"] > 0:
                    summary_list.append({
                        "Method": method_name.replace('_', ' ').title(),
                        "PRS_Set": set_name,
                        "AUC": result["AUC"],
                        "Brier_Score": result.get("Brier_Score", np.nan),
                        "Log_Loss": result.get("Log_Loss", np.nan)
                    })
        
        summary_df = pd.DataFrame(summary_list).sort_values(by="AUC", ascending=False)
        
        # Save results
        if self.output_dir:
            self._save_results(summary_df, pd.DataFrame(all_single_prs_results), stats_comparison, method_results)
        
        return summary_df

    def _plot_calibration_curves(self, method_results: Dict, y_true):
        """
        Creates calibration (reliability) plots for multi-set comparisons.
        """
        GREY_LIGHT = '#D3D3D3'
        
        # Colour palette for different sets
        predefined_colours = ['#FF4D6FFF', '#579EA4FF', '#86AD34FF', '#7E1A2FFF', '#2D2651FF']
        set_colours = {}
        
        for i, set_name in enumerate(self.prs_set_names[:5]):
            set_colours[set_name] = predefined_colours[i]
        
        if len(self.prs_set_names) > 5:
            additional_colours = plt.cm.tab10(np.linspace(0, 1, len(self.prs_set_names)))
            for i, set_name in enumerate(self.prs_set_names[5:], start=5):
                set_colours[set_name] = additional_colours[i]
        
        def style_axes(ax):
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.spines["left"].set_color(GREY_LIGHT)
            ax.spines["left"].set_linewidth(2)
            ax.spines["bottom"].set_color(GREY_LIGHT)
            ax.spines["bottom"].set_linewidth(2)
            ax.tick_params(length=0)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_xlabel('Predicted Probability', size=18, weight='bold')
            ax.set_ylabel('Observed Proportion', size=18, weight='bold')
        
        # Plot for each method: All sets comparison
        for method_name, method_data in method_results.items():
            if not any(res["AUC"] > 0 for res in method_data.values()):
                continue
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            for set_name, result in method_data.items():
                if result["AUC"] > 0:
                    try:
                        fraction_of_positives, mean_predicted_value = calibration_curve(
                            y_true, result["Probs"], n_bins=10, strategy='uniform'
                        )
                        colour = set_colours.get(set_name, '#000000')
                        brier = result.get("Brier_Score", np.nan)
                        ax.plot(mean_predicted_value, fraction_of_positives, 
                               label=f"{set_name} (Brier={brier:.4f})",
                               marker='o', linewidth=2, color=colour)
                    except ValueError as e:
                        print(f"Warning: Could not plot calibration for {set_name}: {e}")
            
            ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Perfectly Calibrated')
            style_axes(ax)
            ax.set_title(f'Calibration Curve: {method_name.replace("_", " ").title()} - All PRS Sets (N={len(y_true):,})', 
                        size=20, weight='bold', pad=20)
            ax.legend(loc="lower right", fontsize=14, frameon=False)
            plt.tight_layout()
            
            if self.output_dir:
                plot_path = os.path.join(self.output_dir, f"calibration_{method_name}_all_sets.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Calibration plot saved: {plot_path}")
            
            plt.show()
            plt.close()
    
    def _plot_decision_curve_analysis(self, method_results: Dict, y_true):
        """
        Creates Decision Curve Analysis plots for multi-set comparisons.
        Net Benefit = TP/N - FP/N * (p_t / (1 - p_t))
        """
        GREY_LIGHT = '#D3D3D3'
        
        # Colour palette for different sets
        predefined_colours = ['#FF4D6FFF', '#579EA4FF', '#86AD34FF', '#7E1A2FFF', '#2D2651FF']
        set_colours = {}
        
        for i, set_name in enumerate(self.prs_set_names[:5]):
            set_colours[set_name] = predefined_colours[i]
        
        if len(self.prs_set_names) > 5:
            additional_colours = plt.cm.tab10(np.linspace(0, 1, len(self.prs_set_names)))
            for i, set_name in enumerate(self.prs_set_names[5:], start=5):
                set_colours[set_name] = additional_colours[i]
        
        def calculate_net_benefit(y_true, probs, threshold):
            """
            Calculate net benefit at a given threshold.
            """
            y_pred = (probs >= threshold).astype(int)
            tn, fp, fn, tp = 0, 0, 0, 0
            
            for true, pred in zip(y_true, y_pred):
                if true == 1 and pred == 1:
                    tp += 1
                elif true == 0 and pred == 1:
                    fp += 1
                elif true == 1 and pred == 0:
                    fn += 1
                else:
                    tn += 1
            
            n = len(y_true)
            if threshold >= 1.0:
                return 0.0
            
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            return net_benefit
        
        def style_axes(ax):
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.spines["left"].set_color(GREY_LIGHT)
            ax.spines["left"].set_linewidth(2)
            ax.spines["bottom"].set_color(GREY_LIGHT)
            ax.spines["bottom"].set_linewidth(2)
            ax.tick_params(length=0)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_xlabel('Threshold Probability', size=18, weight='bold')
            ax.set_ylabel('Net Benefit', size=18, weight='bold')
        
        # Plot for each method: All sets comparison
        for method_name, method_data in method_results.items():
            if not any(res["AUC"] > 0 for res in method_data.values()):
                continue
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            thresholds = np.linspace(0.01, 0.99, 100)
            
            # Calculate net benefit for "treat all" strategy
            prevalence = np.mean(y_true)
            treat_all_nb = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
            ax.plot(thresholds, treat_all_nb, 'k--', alpha=0.5, linewidth=2, label='Treat All')
            
            # Treat none strategy (always 0)
            ax.plot(thresholds, np.zeros_like(thresholds), 'k:', alpha=0.5, linewidth=2, label='Treat None')
            
            for set_name, result in method_data.items():
                if result["AUC"] > 0:
                    net_benefits = [calculate_net_benefit(y_true, result["Probs"], t) for t in thresholds]
                    colour = set_colours.get(set_name, '#000000')
                    ax.plot(thresholds, net_benefits, 
                           label=f"{set_name}",
                           linewidth=2, color=colour)
            
            style_axes(ax)
            ax.set_title(f'Decision Curve Analysis: {method_name.replace("_", " ").title()} - All PRS Sets (N={len(y_true):,})', 
                        size=20, weight='bold', pad=20)
            ax.legend(loc="upper right", fontsize=14, frameon=False)
            ax.set_xlim([0, 1])
            plt.tight_layout()
            
            if self.output_dir:
                plot_path = os.path.join(self.output_dir, f"decision_curve_{method_name}_all_sets.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Decision curve plot saved: {plot_path}")
            
            plt.show()
            plt.close()
    
    def _save_roc_tsv(self, plot_path, curves, y, method_name):
        """Persist (fpr, tpr, threshold, AUC) for every curve drawn next to the PDF."""
        if not plot_path or not curves:
            return
        y_arr = np.asarray(y).astype(int)
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())
        rows = []
        for set_name, res in curves:
            if res is None or res.get("AUC", 0) == 0:
                continue
            probs = np.asarray(res["Probs"], dtype=float)
            fpr, tpr, thr = roc_curve(y_arr, probs)
            for f, t, th in zip(fpr, tpr, thr):
                rows.append({
                    "PRS_Set": set_name,
                    "Method": method_name,
                    "Model_Internal_Name": res.get("Name", set_name),
                    "fpr": f, "tpr": t, "threshold": th,
                    "AUC": float(res["AUC"]),
                    "n_total": int(len(y_arr)),
                    "n_cases": n_pos, "n_controls": n_neg,
                    "Reference_Set": self.reference_set,
                    "Source_Plot": os.path.basename(plot_path),
                })
        if not rows:
            return
        tsv_path = os.path.splitext(plot_path)[0] + ".tsv"
        pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
        print(f"ROC source data saved to: {tsv_path}")

    def _plot_roc_comparisons(self, method_results: Dict, y_true):
        """
        Creates ROC plots for multi-set comparisons.
        """
        GREY_LIGHT = '#D3D3D3'
        
        # Colour palette for different sets
        predefined_colours = ['#FF4D6FFF', '#579EA4FF', '#86AD34FF', '#7E1A2FFF', '#2D2651FF']
        set_colours = {}
        
        # Assign predefined colours to the first 5 sets
        for i, set_name in enumerate(self.prs_set_names[:5]):
            set_colours[set_name] = predefined_colours[i]
        
        # If more than 5 sets, generate additional colours
        if len(self.prs_set_names) > 5:
            additional_colours = plt.cm.tab10(np.linspace(0, 1, len(self.prs_set_names)))
            for i, set_name in enumerate(self.prs_set_names[5:], start=5):
                set_colours[set_name] = additional_colours[i]
        
        def plot_curve(ax, res, label, style='-', colour=None, drawn_curves=None):
            if res["AUC"] == 0:
                return False
            fpr, tpr, _ = roc_curve(y_true, res["Probs"])
            ax.plot(fpr, tpr, label=f"{label} (AUC={res['AUC']:.3f})", linestyle=style, color=colour, linewidth=2)
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
        
        # Plot for each method: All sets comparison
        for method_name, method_data in method_results.items():
            if not any(res["AUC"] > 0 for res in method_data.values()):
                continue
            
            drawn = []
            fig, ax = plt.subplots(figsize=(12, 10))
            
            for set_name, result in method_data.items():
                if result["AUC"] > 0:
                    plot_curve(ax, result, set_name, colour=set_colours.get(set_name), drawn_curves=drawn)
            
            ax.plot([0, 1], [0, 1], 'k:', alpha=0.5)
            style_axes(ax)
            ax.set_title(f'ROC Comparison: {method_name.replace("_", " ").title()} - All PRS Sets (N={len(y_true):,})', 
                        size=20, weight='bold', pad=20)
            ax.legend(loc="lower right", fontsize=14, frameon=False)
            plt.tight_layout()
            
            if self.output_dir:
                plot_path = os.path.join(self.output_dir, f"roc_{method_name}_all_sets.pdf")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
                print(f"ROC plot saved: {plot_path}")
                self._save_roc_tsv(plot_path, drawn, y_true, method_name)
            
            plt.show()
            plt.close()
        
        # Pairwise comparison plots (Reference set vs each other set, for each method)
        for method_name, method_data in method_results.items():
            if not any(res["AUC"] > 0 for res in method_data.values()):
                continue
            
            ref_result = method_data[self.reference_set]
            if ref_result["AUC"] == 0:
                continue
            
            for set_name in self.prs_set_names:
                if set_name == self.reference_set:
                    continue
                
                comp_result = method_data[set_name]
                if comp_result["AUC"] == 0:
                    continue
                
                drawn = []
                fig, ax = plt.subplots(figsize=(10, 10))
                
                plot_curve(ax, ref_result, f"{self.reference_set} (Reference)", 
                          style='--', colour=set_colours[self.reference_set], drawn_curves=drawn)
                plot_curve(ax, comp_result, set_name, colour=set_colours.get(set_name), drawn_curves=drawn)
                
                ax.plot([0, 1], [0, 1], 'k:', alpha=0.5)
                style_axes(ax)
                ax.set_title(f'{method_name.replace("_", " ").title()}: {set_name} vs {self.reference_set} (N={len(y_true):,})',
                           size=20, weight='bold', pad=20)
                ax.legend(loc="lower right", fontsize=14, frameon=False)
                plt.tight_layout()
                
                if self.output_dir:
                    plot_path = os.path.join(self.output_dir, f"roc_{method_name}_{set_name}_vs_{self.reference_set}.pdf")
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
                    print(f"ROC plot saved: {plot_path}")
                    self._save_roc_tsv(plot_path, drawn, y_true, method_name)
                
                plt.show()
                plt.close()

    def _save_results(self, summary_df, all_single_prs_df, stats_comparison_df, method_results):
        print("\nSaving results")
        
        summary_path = os.path.join(self.output_dir, "auc_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"AUC summary saved to: {summary_path}")
        
        # Save all single PRS results
        if self.run_best_single and not all_single_prs_df.empty:
            single_prs_path = os.path.join(self.output_dir, "all_single_prs_results.csv")
            all_single_prs_df.sort_values(by="AUC", ascending=False).to_csv(single_prs_path, index=False)
            print(f"All single PRS results saved to: {single_prs_path}")
        
        if not stats_comparison_df.empty:
            stats_path = os.path.join(self.output_dir, "statistical_comparisons.csv")
            stats_comparison_df.to_csv(stats_path, index=False)
            print(f"Statistical comparisons saved to: {stats_path}")
        
        # Save predicted probabilities
        probs_df = pd.DataFrame({"IID": self.data.index})
        
        for method_name, method_data in method_results.items():
            for set_name, result in method_data.items():
                if result["AUC"] > 0:
                    col_name = f"{method_name}_{set_name}_Probs"
                    probs_df[col_name] = result["Probs"]
        
        probs_df[self.target_col] = self.data[self.target_col].values
        
        probs_path = os.path.join(self.output_dir, "predicted_probabilities.csv")
        probs_df.to_csv(probs_path, index=False)
        print(f"Predicted probabilities saved to: {probs_path}")
        
        # Save detailed method results
        detailed_list = []
        for method_name, method_data in method_results.items():
            for set_name, result in method_data.items():
                if result["AUC"] > 0:
                    detailed_list.append({
                        "Method": method_name.replace('_', ' ').title(),
                        "PRS_Set": set_name,
                        "AUC": result["AUC"],
                        "Brier_Score": result.get("Brier_Score", np.nan),
                        "Log_Loss": result.get("Log_Loss", np.nan),
                        "N_Features": len(self.prs_sets[set_name])
                    })
        
        if detailed_list:
            detailed_df = pd.DataFrame(detailed_list)
            detailed_path = os.path.join(self.output_dir, "detailed_method_results.csv")
            detailed_df.to_csv(detailed_path, index=False)
            print(f"Detailed method results saved to: {detailed_path}")


def load_prs_set(prs_res_root: str, rds_pres_prefix: str, rds_pres_suffix: str,
                 rds_tag_prs: str = None, tag_data: str = None, tag_prs: str = None) -> pd.DataFrame:
    """
    Helper function to load a PRS set with automatic caching.
    
    This function first checks if a cached processed file exists:
    "all_PRS_{rds_pres_prefix}_{rds_pres_suffix}.tsv" in prs_res_root.
    
    If found, it loads from the cached file.
    If not found, it processes individual RDS files and saves the cached file for future use.
    
    :param prs_res_root: Root directory containing individual RDS PRS files
    :param rds_pres_prefix: Prefix for RDS PRS files
    :param rds_pres_suffix: Suffix for RDS PRS files
    :param rds_tag_prs: Tag for PRS model type
    :param tag_data: Tag for data file
    :param tag_prs: Tag for PRS prediction
    :return: DataFrame with PRS values indexed by IID
    """
    # Construct cached file path
    # Remove leading/trailing underscores from prefix/suffix for clean filename
    clean_prefix = rds_pres_prefix.strip('_')
    clean_suffix = rds_pres_suffix.strip('_').replace('.rds', '')
    cached_filename = f"all_PRS_{clean_prefix}_{clean_suffix}.tsv"
    cached_filepath = os.path.join(prs_res_root, cached_filename)
    
    # Check if cached file exists
    if os.path.exists(cached_filepath):
        print(f"Found cached PRS file: {cached_filename}")
        print(f"Loading from: {cached_filepath}")
        prs = pd.read_table(cached_filepath, sep="\t")
        prs.set_index("IID", inplace=True)
        print(f"Loaded {len(prs)} samples with {len(prs.columns)} PRS features from cache")
        return prs
    
    # If cached file doesn't exist, process individual RDS files
    print(f"No cached file found. Processing individual PRS files from: {prs_res_root}")
    pths_prs = glob(f"{prs_res_root}/{rds_pres_prefix}*{rds_pres_suffix}")
    
    if not pths_prs:
        raise FileNotFoundError(f"No PRS files found matching pattern: {prs_res_root}/{rds_pres_prefix}*{rds_pres_suffix}")
    
    print(f"Found {len(pths_prs)} RDS files to process...")
    prs_store = []
    for pth_prs in tqdm(pths_prs, desc="Processing RDS files"):
        latent = pth_prs.split(rds_pres_prefix)[1].split(rds_pres_suffix)[0]
        
        data = pyreadr.read_r(pth_prs.replace(rds_tag_prs, tag_data))[None]
        prs_latent = data[['IID']].copy()
        r_obj = read_rds(pth_prs)
        prs_latent[f"PRS:{latent}"] = r_obj[tag_prs].copy()
        prs_latent.set_index("IID", inplace=True)
        prs_store.append(prs_latent)
    
    prs = pd.concat(prs_store, axis=1)
    print(f"Processed {len(prs_store)} PRS files -> {len(prs)} samples with {len(prs.columns)} features")
    
    # Save cached file for future use
    print(f"Saving cached PRS file: {cached_filename}")
    prs_to_save = prs.reset_index()
    prs_to_save.to_csv(cached_filepath, sep="\t", index=False)
    print(f"Cached file saved to: {cached_filepath}")
    
    return prs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-PRS Set Comparison Pipeline - Compare multiple PRS sets against each other')
    
    # Common file paths
    parser.add_argument('--pth_covars', type=str, default="/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/covars/nonDisc_caucasian_king0p0625_V0.tsv",
                        help='Path to covariates TSV file')
    parser.add_argument('--pth_dis', type=str, default="/project/ukbblatent/clinicaldata/binary_disease_cohorts/F20208v3_nonDiscov/caucasian_king0p0625_grouped/newcovsets/V0v2/atherosclerotic.csv",
                        help='Path to disease cohort CSV file')
    parser.add_argument('--pth_out_root', type=str, default="/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/comparisons",
                        help='Output root directory for results')
    parser.add_argument('--pth_out_dir', type=str, default="IDPPRS_athero_V0v2",
                        help='Output directory name (param names will be concatenated)')
    
    # PRS Set arguments (repeatable)
    parser.add_argument('--prs_set_name', type=str, action='append', required=True,
                        help='Name for a PRS set (repeatable, e.g., --prs_set_name My_Custom --prs_set_name LDPred2)')
    parser.add_argument('--prs_set_root', type=str, action='append', required=True,
                        help='Root directory for a PRS set containing RDS files or cached TSV (repeatable, corresponds to prs_set_name order)')
    
    # RDS file parameters (used for processing and caching)
    parser.add_argument('--rds_pres_prefix', type=str, action='append',
                        help='Prefix for RDS PRS files (repeatable, corresponds to prs_set_name order)')
    parser.add_argument('--rds_pres_suffix', type=str, action='append',
                        help='Suffix for RDS PRS files (repeatable, corresponds to prs_set_name order)')
    parser.add_argument('--rds_tag_prs', type=str, default='auto.mod',
                        help='Tag for PRS model type (applied to all sets if not individually specified)')
    parser.add_argument('--tag_data', type=str, default='resNdata.basic',
                        help='Tag for data file (applied to all sets)')
    parser.add_argument('--tag_prs', type=str, default='pred_auto',
                        help='Tag for PRS prediction (applied to all sets)')
    
    # MultiPRSSetEvaluator parameters
    parser.add_argument('--target_col', type=str, default='BinCAT_Disease',
                        help='Name of the binary disease outcome column')
    parser.add_argument('--covariates', type=str, nargs='+', default=['Age', 'BMI'],
                        help='List of continuous covariate column names')
    parser.add_argument('--cat_covariates', type=str, nargs='+', default=['Sex', 'CAT_Smoking'],
                        help='List of categorical covariate column names')
    parser.add_argument('--reference_set', type=str, default=None,
                        help='Name of reference PRS set for comparisons (defaults to first set)')
    parser.add_argument('--ignore_features', type=str, nargs='+', default=["PRS:LV_longitudinal_strain_Segment_3","PRS:LV_longitudinal_strain_Segment_6","PRS:LV_longitudinal_strain_Segment_5"],
                        help='List of feature names (original, unprefixed) to exclude from all PRS sets')
    parser.add_argument('--orthogonalise', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to orthogonalise PRS against covariates (removes covariate effects from PRS)')
    parser.add_argument('--use_covariates_as_inputs', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to include covariates as input variables in models (works independently from orthogonalise)')
    parser.add_argument('--use_variational_inference', action=argparse.BooleanOptionalAction, default=False,
                        help='Use Variational Inference (ADVI) instead of MCMC for Bayesian methods')
    
    # Method selection flags
    parser.add_argument('--run_best_single', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to evaluate best single PRS for each set')
    parser.add_argument('--run_elastic_net', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to run Elastic Net method')
    parser.add_argument('--run_pca', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to run PCA method')
    parser.add_argument('--run_bayesian_pymc', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to run Bayesian Hierarchical (PyMC) method')
    parser.add_argument('--run_bayesian_bambi', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to run Bayesian Hierarchical (Bambi) method')
    
    args = parser.parse_args()
    
    # Validate PRS set inputs
    if not args.prs_set_name:
        raise ValueError("Must provide at least one --prs_set_name")
    
    if len(args.prs_set_root) != len(args.prs_set_name):
        raise ValueError("Number of --prs_set_root must match --prs_set_name")
    
    # Load covariates
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    covar = pd.read_table(args.pth_covars, sep="\t")
    covar.set_index("IID", inplace=True)
    print(f"Loaded covariates: {len(covar)} samples")
    
    # Load disease outcomes
    disDF = pd.read_csv(args.pth_dis)
    disDF.set_index("eid", inplace=True)
    disDF = disDF[[args.target_col]]
    print(f"Loaded disease outcomes: {len(disDF)} samples")
    
    # Load all PRS sets
    prs_sets_dict = {}
    
    for i, set_name in enumerate(args.prs_set_name):
        print(f"\nLoading PRS set: {set_name}")
        
        prs_root = args.prs_set_root[i]
        
        # Get RDS parameters with defaults
        prefix = args.rds_pres_prefix[i] if args.rds_pres_prefix and len(args.rds_pres_prefix) > i else "run_ext_basic_king0p0625_lw_gw_indep_FiltMAF_"
        suffix = args.rds_pres_suffix[i] if args.rds_pres_suffix and len(args.rds_pres_suffix) > i else ".fullDS.auto.mod.LDPred2.rds"
        
        # Load PRS with automatic caching
        prs_df = load_prs_set(
            prs_res_root=prs_root,
            rds_pres_prefix=prefix,
            rds_pres_suffix=suffix,
            rds_tag_prs=args.rds_tag_prs,
            tag_data=args.tag_data,
            tag_prs=args.tag_prs
        )
        
        prs_sets_dict[set_name] = prs_df

    pth_out_dir = args.pth_out_dir
    pth_out_dir += "_ortho" if args.orthogonalise else "_noortho"
    pth_out_dir += "_covarinput" if args.use_covariates_as_inputs else "_nocovarinput"
    pth_out_dir += "_vi" if args.use_variational_inference else "_mcmc"
    full_output_dir = os.path.join(args.pth_out_root, pth_out_dir)
    
    # Run comparison
    print("\n" + "="*80)
    print("INITIALISING EVALUATOR")
    print("="*80)
    
    inference_mode = "Variational Inference (ADVI)" if args.use_variational_inference else "MCMC (NUTS)"
    print(f"Settings: Orthogonalisation={args.orthogonalise}, Covariates as Inputs={args.use_covariates_as_inputs}, Bayesian Inference={inference_mode}")
    
    # Display virtual subgroups if configured
    if VIRTUAL_SUBGROUPS:
        print(f"\nVirtual Subgroups Configured: {len(VIRTUAL_SUBGROUPS)}")
        for subgroup_name, features in VIRTUAL_SUBGROUPS.items():
            print(f"  - {subgroup_name}: {len(features)} features")
    
    evaluator = MultiPRSSetEvaluator(
        covar, prs_sets_dict, disDF,
        target_col=args.target_col,
        covariates=args.covariates,
        cat_covariates=args.cat_covariates,
        reference_set=args.reference_set,
        virtual_subgroups=VIRTUAL_SUBGROUPS if VIRTUAL_SUBGROUPS else None,
        ignore_features=args.ignore_features,
        orthogonalise=args.orthogonalise,
        use_covariates_as_inputs=args.use_covariates_as_inputs,
        output_dir=full_output_dir,
        use_variational_inference=args.use_variational_inference,
        run_best_single=args.run_best_single,
        run_elastic_net=args.run_elastic_net,
        run_pca=args.run_pca,
        run_bayesian_pymc=args.run_bayesian_pymc,
        run_bayesian_bambi=args.run_bayesian_bambi
    )
    
    summary = evaluator.run_comparison()
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(summary)
