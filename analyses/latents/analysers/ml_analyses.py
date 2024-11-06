# Enabling Intel Extension for Scikit-learn (comment it out if not required/working)
#from sklearnex import patch_sklearn
#patch_sklearn()

import sys
import argparse
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from functools import partial
import statsmodels.api as sm
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, LassoCV, Lasso
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import stats
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import pickle

sys.path.insert(0, os.getcwd()) #to handle the sub-foldered structure of the tricoder

from data_handler import AnalysersDataHandler
from analyses.latents.utils import remove_high_vif_features

thread_local = threading.local()

def get_analyser_copy(analyser):
    if not hasattr(thread_local, 'analyser'):
        thread_local.analyser = deepcopy(analyser) 
    return thread_local.analyser

def process_attribute(target_attribute, analyser, args, result_queue):
    analyser = get_analyser_copy(analyser)
    X = analyser.X.copy()
    X_test = analyser.X_heldout if analyser.X_heldout is None else analyser.X_heldout.copy()
    y = analyser.y[target_attribute].copy()
    y_test = analyser.y_heldout if analyser.y_heldout is None else analyser.y_heldout[target_attribute].copy()

    combined = pd.concat([X, y], axis=1).dropna()
    X, y = combined[X.columns], combined[y.name]

    if analyser.X_heldout is not None:
        combined = pd.concat([X_test, y_test], axis=1).dropna()
        X_test, y_test = combined[X_test.columns], combined[y_test.name]

    if bool(analyser.cov_cols):
        print('\n--- Association analysis-------\n')
        combined = analyser.df_association.join(y, how="inner").dropna()
        X_assoc, y_assoc = combined[analyser.df_association.columns], combined[y.name]
        analyser.association(
            X_train=analyser.scale_features(X_train=X_assoc, X_test=None)[0] if args.use_feature_scaling else X_assoc,
            y_train=analyser.scale_target(y_train=y_assoc, y_test=None)[0] if args.use_target_scaling else y_assoc,
            exp_type=f"{target_attribute}_Assoc_dataset",
            level_type="Train"
        )

    if X_test is not None or args.run_also_trainonly:
        if args.use_feature_scaling:
            X, X_test = analyser.scale_features(X_train=X, X_test=X_test)
        if args.use_target_scaling and (target_attribute in analyser.continuous_attributes):
            y, y_test = analyser.scale_target(y_train=y, y_test=y_test)
        if args.remove_with_VIF:
            print('\n--- Remove with VIF, full dataset---')
            X = remove_high_vif_features(X, threshold=10)
            X_test = X_test if X_test is None else X_test[X.columns]
            analyser.latent_factors = X.columns
        analyser.train_test(
            X_train=X, X_test=X_test, y_train=y, y_test=y_test,
            target_attribute=target_attribute, exp_type="dataset",
            level_type=("heldoutTest" if X_test is not None else "Train")
        )

    kf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=1701) if target_attribute in analyser.categorical_attributes else KFold(n_splits=args.n_folds, shuffle=True, random_state=1701)
    for fold, (train_index, test_index) in enumerate(kf.split(np.zeros(len(y)) if target_attribute in analyser.categorical_attributes else X, y)):
        X_train, X_test, y_train, y_test = X.iloc[train_index, :], X.iloc[test_index, :], y.iloc[train_index], y.iloc[test_index]

        if args.use_feature_scaling:
            X_train, X_test = analyser.scale_features(X_train=X_train, X_test=X_test)
        if args.use_target_scaling and (target_attribute in analyser.continuous_attributes):
            y_train, y_test = analyser.scale_target(y_train=y_train, y_test=y_test)
        if args.remove_with_VIF:
            print(f'\n--- Remove with VIF, fold {fold}')
            X_train = remove_high_vif_features(X_train, threshold=10)
            X_test = X_test[X_train.columns]
            analyser.latent_factors = X_train.columns

        analyser.train_test(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            target_attribute=target_attribute, exp_type="CV",
            level_type=f"Fold_{fold}"
        )
    
    # Collect the results
    result_queue.put((target_attribute, analyser))

def parallel_process_attributes(analyser, args, num_threads):
    result_queue = Queue()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_attribute, target_attribute, analyser, args, result_queue): target_attribute for target_attribute in analyser.attributes}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Attributes: "):
            target_attribute = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {target_attribute}: {e}")

    # Aggregate results back into the main analyser
    while not result_queue.empty():
        target_attribute, partial_analyser = result_queue.get()
        analyser.merge_results(partial_analyser, target_attribute)

class MLAnalyses(AnalysersDataHandler):
    def __init__(self, args):
        super().__init__(args)

        if bool(self.cov_cols):
            self.df_association = self.df[self.cov_cols + self.latent_factors]
        
        self.X = self.df[self.latent_factors]
        self.X_heldout = self.df_test if self.df_test is None else self.df_test[self.latent_factors]
        self.y = self.df[self.attributes]
        self.y_heldout = self.df_test if self.df_test is None else self.df_test[self.attributes]

        # Scalers and Remove with VIF now moved in main(): fit on train data, while test data gets only transformed
        if self.use_feature_scaling:
            #scaler = StandardScaler()
            #X_scaled = scaler.fit_transform(self.X)
            #self.X = pd.DataFrame(X_scaled, columns=self.latent_factors, index=self.df.index)
            self.tsv_name = f"FScale_{self.tsv_name}"

        if self.use_target_scaling:
            #scaler = StandardScaler()
            #y_scaled = scaler.fit_transform(self.y)
            #self.y = pd.DataFrame(y_scaled, columns=self.attributes, index=self.df.index)
            self.tsv_name = f"TScale_{self.tsv_name}"

        if self.remove_with_VIF:
            #self.X = remove_high_vif_features(self.X, threshold=10)
            #self.latent_factors = self.X.columns
            self.tsv_name = f"VIF_{self.tsv_name}"

    def merge_results(self, partial_analyser, target_attribute):
        self.res_collect.update(partial_analyser.res_collect)
        # Save the partial analyser as a pickle file
        # filename = f'partial_result_{target_attribute}.pkl'
        # with open(filename, 'wb') as f:
        #     pickle.dump(partial_analyser, f)
        # print(f"Partial results for {target_attribute} saved to {filename}")

    ####### Eval functions (Start) #######
    def eval_linear(self, model, features, X_train, y_train, X_test=None, y_test=None, exp_type="", level_type=""):
        if X_test is not None:
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            if bool(self.out_path) and bool(exp_type):
                self.res_collect[exp_type][level_type]['MSE_TestSet'] = mse
                self.res_collect[exp_type][level_type]['R-squared_TestSet'] = r2
            else:
                print(f'Mean Squared Error (Test Set): {mse}')
                print(f'R-squared (Test Set): {r2}')

        coefficients = pd.DataFrame(model.coef_, index=features, columns=['Coefficient'])
        intercept = model.intercept_
        
        n = len(X_train)
        k = len(features)
        dof = n - k - 1
        residuals = y_train - model.predict(X_train)
        mse = np.sum(residuals**2) / dof

        t_values, p_values = [], []
        for i, coef in enumerate(coefficients['Coefficient']):
            standard_error = np.sqrt(mse / np.sum((X_train.iloc[:, i] - X_train.iloc[:, i].mean())**2))
            t_stat = coef / standard_error
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))
            t_values.append(t_stat)
            p_values.append(p_value)

        coefficients['t-value'] = t_values
        coefficients['p-value'] = p_values

        if bool(self.out_path) and bool(exp_type):
            self.res_collect[exp_type][level_type]['TrainSize'] = len(X_train)
            self.res_collect[exp_type][level_type]['TrainParticipants'] = X_train.index
            if X_test is not None:
                self.res_collect[exp_type][level_type]['TestSize'] = len(X_test)
                self.res_collect[exp_type][level_type]['TestParticipants'] = X_test.index
            self.res_collect[exp_type][level_type]['Coefficients'] = coefficients
            self.res_collect[exp_type][level_type]['Intercept'] = intercept
        else:
            print(f'Train size: {len(X_train)}')
            if X_test is not None:
                print(f'Test size: {len(X_test)}')
            print("Coefficients with t-values and p-values:")
            print(coefficients)
            print(f'Intercept: {intercept}')

        significance_level = 0.05
        significant_coefficients = coefficients[coefficients['p-value'] < significance_level]

        if bool(self.out_path) and bool(exp_type):
            self.res_collect[exp_type][level_type]['SignificantCoefficients'] = significant_coefficients
        else:
            print("\nSignificant coefficients:")
            print(significant_coefficients)


    def eval_logistic(self, model, features, X_train, X_test=None, y_test=None, is_binary=False, exp_type="", level_type=""):
        if X_test is not None:
            y_pred = model.predict(X_test)
            if bool(self.out_path) and bool(exp_type):
                df_rs = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
                self.res_collect[exp_type][level_type]['ClassifRprt_TestSet'] = df_rs
            else:
                print("Classification report:")
                print(classification_report(y_test, y_pred))

        classes = ["Coefficient_BinaryClassify"] if is_binary else [f"Coefficient_class_{str(c)}" for c in model.classes_]
        coefficients = pd.DataFrame(model.coef_.T, index=features, columns=classes)
        intercept = model.intercept_

        for cls in classes:
            coefficients[cls.replace("Coefficient", "OddsRatio")] = np.exp(coefficients[cls]) # Calculate odds ratios

            # Calculate the standard errors
            X_std = X_train.std()
            standard_errors = X_std * np.sqrt(np.diag(np.linalg.inv(np.dot(X_train.T, X_train))))

            # Calculate the confidence intervals
            z = stats.norm.ppf(1 - 0.05 / 2)
            coefficients[cls.replace("Coefficient", "LowerCI")] = np.exp(coefficients[cls] - z * standard_errors)
            coefficients[cls.replace("Coefficient", "UpperCI")] = np.exp(coefficients[cls] + z * standard_errors)

        if bool(self.out_path) and bool(exp_type):
            self.res_collect[exp_type][level_type]['TrainSize'] = len(X_train)
            self.res_collect[exp_type][level_type]['TrainParticipants'] = X_train.index
            if X_test is not None:
                self.res_collect[exp_type][level_type]['TestSize'] = len(X_test)
                self.res_collect[exp_type][level_type]['TestParticipants'] = X_test.index
            self.res_collect[exp_type][level_type]['Coefficients'] = coefficients
            self.res_collect[exp_type][level_type]['Intercept'] = intercept
        else:
            print(f'Train size: {len(X_train)}')
            if X_test is not None:
                print(f'Test size: {len(X_test)}')
            print("Odds Ratios and Confidence Intervals:")
            print(coefficients)
            print(f'Intercept: {intercept}')

        for cls in classes:
            # Identify significant coefficients based on confidence intervals
            significant_coefficients = coefficients[(coefficients[cls.replace("Coefficient", "LowerCI")] > 1) | (coefficients[cls.replace("Coefficient", "UpperCI")] < 1)]
            
            if bool(self.out_path) and bool(exp_type):
                self.res_collect[exp_type][level_type][f'SignificantCoefficients_{cls}'] = significant_coefficients
            else:
                print(f"\nSignificant coefficients for {cls}:")
                print(significant_coefficients)

    
    def eval_association(self, model, feature, X, y, exp_type="", level_type=""):
        effect_size = model.params[feature]
        std_error = model.bse[feature]
        p_value = model.pvalues[feature]

        if bool(self.out_path) and bool(exp_type):
            self.res_collect[exp_type][level_type]['TrainSize'] = len(X)
            self.res_collect[exp_type][level_type]['TrainParticipants'] = X.index
            self.res_collect[exp_type][level_type]['EffectSize'] = effect_size
            self.res_collect[exp_type][level_type]['StdError'] = std_error
            self.res_collect[exp_type][level_type]['p-value'] = p_value
        else:
            print(f'Train size: {len(X)}')
            print(f'Effect size {level_type}: {effect_size}')
            print(f'Std Error {level_type}: {std_error}')
            print(f'p-value {level_type}: {p_value}')
               
    ####### Eval functions (End) #######


    ##### Single model functions (Start) #####

    def single_linear_regression(self, X_train, X_test, y_train, y_test, exp_type, level_type):
        model = LinearRegression(n_jobs=self.n_jobs)
        model.fit(X_train, y_train)

        self.eval_linear(model=model, features=self.latent_factors, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, exp_type=exp_type, level_type=level_type)

    def single_logistic_regression(self, X_train, X_test, y_train, y_test, is_binary, exp_type, level_type):
        model = LogisticRegression(max_iter=self.max_iter, solver="saga", multi_class='ovr' if is_binary else 'multinomial', n_jobs=self.n_jobs)
        model.fit(X_train, y_train)
        
        self.eval_logistic(model=model, features=self.latent_factors, X_train=X_train, X_test=X_test, y_test=y_test, is_binary=is_binary, exp_type=exp_type, level_type=level_type)

    ####### Single model functions (End) #######

    #### Association function (Start) ####

    def association(self, X_train, y_train, exp_type, level_type):
        for latent_factor in tqdm(self.latent_factors):  # loop through all the latent factors
            X = X_train[self.cov_cols].join(y_train, how="inner", rsuffix="_")
            if isinstance(y_train.iloc[0], str):
                X[y_train.name] = pd.factorize(X[y_train.name])[0]
            X = sm.add_constant(X)
            y = X_train[latent_factor]
            
            model = sm.OLS(y, X).fit()

            self.eval_association(model=model, feature=y_train.name, X=X, y=y, exp_type=exp_type, level_type=y.name)

    #### Association function (End) ####

    ##### RFE model functions (Start) #####

    def RFE_linaer_regression(self, X_train, X_test, y_train, y_test, exp_type, level_type):
        for n_feat in tqdm(range(2, len(self.latent_factors))):
            model = LinearRegression(n_jobs=self.n_jobs)

            # Perform RFE to find the most important features
            selector = RFE(model, n_features_to_select=n_feat, step=1) 
            selector = selector.fit(X_train, y_train)

            # Display the selected features
            selected_features = [self.latent_factors[i] for i, mask in enumerate(selector.support_) if mask]
            if bool(self.out_path) and bool(exp_type):
                self.res_collect[f"{exp_type}_nfeat{n_feat}"][level_type]['SelectedFeatures'] = selected_features
            else:
                print("Selected features:", selected_features)

            # Train the model with the selected features
            X_selected = X_train[selected_features]
            model.fit(X_selected, y_train)
            
            self.eval_linear(model=model, features=selected_features, X_train=X_selected, y_train=y_train, X_test=X_test[selected_features] if X_test is not None else None, y_test=y_test, exp_type=f"{exp_type}_nfeat{n_feat}" if bool(exp_type) else "", level_type=level_type)


    def RFE_logistic_regression(self, X_train, X_test, y_train, y_test, is_binary, exp_type, level_type):
        for n_feat in tqdm(range(2, len(self.latent_factors))):
            model = LogisticRegression(max_iter=self.max_iter, solver="saga", multi_class='ovr' if is_binary else 'multinomial', n_jobs=self.n_jobs)

            # Perform RFE to find the most important features
            selector = RFE(model, n_features_to_select=n_feat, step=1) 
            selector = selector.fit(X_train, y_train)

            # Display the selected features
            selected_features = [self.latent_factors[i] for i, mask in enumerate(selector.support_) if mask]
            if bool(self.out_path) and bool(exp_type):
                self.res_collect[f"{exp_type}_nfeat{n_feat}"][level_type]['SelectedFeatures'] = selected_features
            else:
                print("Selected features:", selected_features)

            # Train the model with the selected features
            X_selected = X_train[selected_features]
            model.fit(X_selected, y_train)

            self.eval_logistic(model=model, features=selected_features, X_train=X_selected, X_test=X_test[selected_features] if X_test is not None else None, y_test=y_test, is_binary=is_binary, exp_type=f"{exp_type}_nfeat{n_feat}" if bool(exp_type) else "", level_type=level_type)

    ####### RFE model functions (End) #######

    ##### Lasso functions (Start) #####

    def lasso(self, X_train, X_test, y_train, y_test, exp_type, level_type, l1_pen=None):
        if l1_pen is not None:
            optimal_alpha = l1_pen
        else:
            lasso_cv = LassoCV(alphas=None, cv=5, max_iter=self.max_iter_CV, random_state=1701, n_jobs=self.n_jobs)
            lasso_cv.fit(X_train, y_train)
            optimal_alpha = lasso_cv.alpha_
        if bool(self.out_path) and bool(exp_type):
            self.res_collect[exp_type][level_type]['TrainSize'] = len(X_train)
            self.res_collect[exp_type][level_type]['TrainParticipants'] = X_train.index
            if X_test is not None:
                self.res_collect[exp_type][level_type]['TestSize'] = len(X_test)
                self.res_collect[exp_type][level_type]['TestParticipants'] = X_test.index
            self.res_collect[exp_type][level_type]['OptimalAlpha'] = optimal_alpha
        else:
            print(f'Train size: {len(X_train)}')
            if X_test is not None:
                print(f'Test size: {len(X_test)}')
            print(f"Optimal alpha value: {optimal_alpha}")

        model = Lasso(alpha=optimal_alpha, max_iter=self.max_iter, random_state=1701)
        model.fit(X_train, y_train)

        if X_test is not None:
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            if bool(self.out_path) and bool(exp_type):
                self.res_collect[exp_type][level_type]['MSE_TestSet'] = mse
                self.res_collect[exp_type][level_type]['R-squared_TestSet'] = r2
            else:
                print(f'Mean Squared Error (Test Set): {mse}')
                print(f'R-squared (Test Set): {r2}')

        coefficients = pd.DataFrame(model.coef_, index=self.latent_factors, columns=['Coefficient'])
        intercept = model.intercept_

        if bool(self.out_path) and bool(exp_type):
            self.res_collect[exp_type][level_type]['Coefficients'] = coefficients
            self.res_collect[exp_type][level_type]['Intercept'] = intercept
        else:
            print("Coefficients:")
            print(coefficients)
            print(f'Intercept: {intercept}')

        significant_coefficients = coefficients[coefficients['Coefficient'] != 0]

        if bool(self.out_path) and bool(exp_type):
            self.res_collect[exp_type][level_type]['SignificantCoefficients'] = significant_coefficients
        else:
            print("\nSignificant coefficients:")
            print(significant_coefficients)

    def lasso_categorical(self, X_train, X_test, y_train, y_test, is_binary, exp_type, level_type, l1_pen=None):
        # LogisticRegressionCV model with L1 regularization -  essentially the logistic regression equivalent of Lasso.
        if l1_pen is not None:
            optimal_C = l1_pen
        else:
            logreg_cv = LogisticRegressionCV(Cs=10, cv=5, penalty='l1', solver='saga', max_iter=self.max_iter_CV, random_state=1701, multi_class='ovr' if is_binary else 'multinomial', n_jobs=self.n_jobs)
            logreg_cv.fit(X_train, y_train)
            optimal_C = logreg_cv.C_[0]
        if bool(self.out_path) and bool(exp_type):
            self.res_collect[exp_type][level_type]['OptimalC'] = optimal_C
        else:
            print(f"Optimal C value: {optimal_C}")

        model = LogisticRegression(C=optimal_C, penalty='l1', solver='saga', max_iter=self.max_iter, random_state=1701, multi_class='ovr' if is_binary else 'multinomial', n_jobs=self.n_jobs)
        model.fit(X_train, y_train)

        self.eval_logistic(model=model, features=self.latent_factors, X_train=X_train, X_test=X_test, y_test=y_test, is_binary=is_binary, exp_type=exp_type, level_type=level_type)

    ##### Lasso functions (End) #####


    ##### SelectFromModel functions (Start) #####
    
    def linear_regression_withFeatureSelection(self, X_train, X_test, y_train, y_test, exp_type, level_type):
        selector = SelectFromModel(estimator=LinearRegression(n_jobs=self.n_jobs)).fit(X_train, y_train)
        selected_features = [self.latent_factors[i] for i, mask in enumerate(selector.get_support()) if mask]        
        if bool(self.out_path) and bool(exp_type):
            self.res_collect[f"{exp_type}_FeatureSelection"][level_type]['SelectedFeatures'] = selected_features
        else:
            print("Selected features:", selected_features)

        # Train the model with the selected features
        X_selected = X_train[selected_features]
        model  = LinearRegression(n_jobs=self.n_jobs)
        model.fit(X_selected, y_train)
        
        self.eval_linear(model=model, features=selected_features, X_train=X_selected, y_train=y_train, X_test=X_test[selected_features] if X_test is not None else None, y_test=y_test, exp_type=f"{exp_type}_FeatureSelection" if bool(exp_type) else "", level_type=level_type)

    def logistic_regression_withFeatureSelection(self, X_train, X_test, y_train, y_test, is_binary, exp_type, level_type):
        selector = SelectFromModel(estimator=LogisticRegression(max_iter=self.max_iter, solver="saga", multi_class='ovr' if is_binary else 'multinomial', n_jobs=self.n_jobs)).fit(X_train, y_train)
        selected_features = [self.latent_factors[i] for i, mask in enumerate(selector.get_support()) if mask]
        if bool(self.out_path) and bool(exp_type):
            self.res_collect[f"{exp_type}_FeatureSelection"][level_type]['SelectedFeatures'] = selected_features
        else:
            print("Selected features:", selected_features)

        # Train the model with the selected features
        X_selected = X_train[selected_features]
        model = LogisticRegression(max_iter=self.max_iter, solver="saga", multi_class='ovr' if is_binary else 'multinomial', n_jobs=self.n_jobs)
        model.fit(X_selected, y_train)

        self.eval_logistic(model=model, features=selected_features, X_train=X_selected, X_test=X_test[selected_features] if X_test is not None else None, y_test=y_test, is_binary=is_binary, exp_type=f"{exp_type}_FeatureSelection" if bool(exp_type) else "", level_type=level_type)

    ##### SelectFromModel functions (End) #####

    def save_results(self, flush_results=False):
        if bool(self.out_path):
            os.makedirs(self.out_path, exist_ok=True)
            with open(os.path.join(self.out_path, f"{self.tsv_name}_results.pkl"), "wb") as f:
                pickle.dump(self.res_collect, f)
            if flush_results:
                self.res_collect = defaultdict(partial(defaultdict, dict)) 

    def train_test(self, X_train, X_test, y_train, y_test, target_attribute, exp_type, level_type):
        if self.df_add is not None:  # prediction with additional features is required
            df_add_train = self.df_add.loc[list(set(self.df_add.index).intersection(set(X_train.index))), :]  # train participants
            df_add_test = self.df_add.loc[list(set(self.df_add.index).intersection(set(X_test.index))), :]  # test participants
            if self.use_feature_scaling:  # scale additional features
                df_add_train, df_add_test = self.scale_features(X_train=df_add_train, X_test=df_add_test)
            X_train_add, X_test_add = df_add_train.join(X_train, how="inner"), df_add_test.join(X_test, how="inner")  # full X, with also additional features
            y_train_add, y_test_add = df_add_train.join(y_train, how="inner", rsuffix="_")[target_attribute], df_add_test.join(y_test, how="inner", rsuffix="_")[target_attribute]  # corresponding y

        if target_attribute in self.categorical_attributes: #It's categorical, run Logistic Regression
            self.single_logistic_regression(X_train, X_test, y_train, y_test, is_binary=target_attribute in self.binary_attributes, exp_type=f"{target_attribute}_LogRegrs_{exp_type}", level_type=level_type)
            # self.logistic_regression_withFeatureSelection(X_train, X_test, y_train, y_test, is_binary=target_attribute in self.binary_attributes, exp_type=f"{target_attribute}_LogRegrs_wFS_{exp_type}", level_type=level_type)
            self.RFE_logistic_regression(X_train, X_test, y_train, y_test, is_binary=target_attribute in self.binary_attributes, exp_type=f"{target_attribute}_RFELogRegrs_{exp_type}", level_type=level_type)
            self.lasso_categorical(X_train, X_test, y_train, y_test, is_binary=target_attribute in self.binary_attributes, exp_type=f"{target_attribute}_LassoCat_{exp_type}", level_type=level_type)
            if self.l1_penalty > 0:
                self.lasso_categorical(X_train, X_test, y_train, y_test, l1_pen=self.l1_penalty, is_binary=target_attribute in self.binary_attributes, exp_type=f"{target_attribute}_LassoCat_customL1_{exp_type}", level_type=level_type)
            if self.df_add is not None:  # prediction with additional features is required
                self.latent_factors = list(X_train_add.columns)  # new predictors (with also additional features)
                self.single_logistic_regression(X_train_add, X_test_add, y_train_add, y_test_add, is_binary=target_attribute in self.binary_attributes, exp_type=f"{target_attribute}_LogRegrs_AddFeat_{exp_type}", level_type=level_type)
                self.lasso_categorical(X_train_add, X_test_add, y_train_add, y_test_add, is_binary=target_attribute in self.binary_attributes, exp_type=f"{target_attribute}_LassoCat_AddFeat_{exp_type}", level_type=level_type)
                if self.l1_penalty > 0:
                    self.lasso_categorical(X_train_add, X_test_add, y_train_add, y_test_add, l1_pen=self.l1_penalty, is_binary=target_attribute in self.binary_attributes, exp_type=f"{target_attribute}_LassoCat_customL1_AddFeat_{exp_type}", level_type=level_type)
                self.latent_factors = X_train.columns

        else:
            self.single_linear_regression(X_train, X_test, y_train, y_test, exp_type=f"{target_attribute}_LinRegrs_{exp_type}", level_type=level_type)
            # self.linear_regression_withFeatureSelection(X_train, X_test, y_train, y_test, exp_type=f"{target_attribute}_LinRegrs_wFS_{exp_type}", level_type=level_type)
            self.RFE_linaer_regression(X_train, X_test, y_train, y_test, exp_type=f"{target_attribute}_RFELinRegrs_{exp_type}", level_type=level_type)
            self.lasso(X_train, X_test, y_train, y_test, exp_type=f"{target_attribute}_Lasso_{exp_type}", level_type=level_type)
            if self.l1_penalty > 0:
                self.lasso(X_train, X_test, y_train, y_test, l1_pen=self.l1_penalty, exp_type=f"{target_attribute}_Lasso_customL1_{exp_type}", level_type=level_type)
            if self.df_add is not None:  # prediction with additional features is required
                self.latent_factors = list(X_train_add.columns)  # new predictors (with also additional features)
                self.single_linear_regression(X_train_add, X_test_add, y_train_add, y_test_add, exp_type=f"{target_attribute}_LinRegrs_AddFeat_{exp_type}", level_type=level_type)
                self.lasso(X_train_add, X_test_add, y_train_add, y_test_add, exp_type=f"{target_attribute}_Lasso_AddFeat_{exp_type}", level_type=level_type)
                if self.l1_penalty > 0:
                    self.lasso(X_train_add, X_test_add, y_train_add, y_test_add, l1_pen=self.l1_penalty, exp_type=f"{target_attribute}_Lasso_customL1_AddFeat_{exp_type}", level_type=level_type)
                self.latent_factors = X_train.columns

    def scale_features(self, X_train, X_test):
        col_std = [x for x in X_train.columns if not(x.startswith('BinCAT_') or x.startswith('CAT_'))]  # scale only continuous features
        ct = ColumnTransformer(transformers=[('std_scaler', StandardScaler(), col_std)], remainder='passthrough', sparse_threshold=0, verbose_feature_names_out=False)  # standardize each column independently (x_std = (x - mean) / std_dev)
        X = pd.DataFrame(ct.fit_transform(X_train), columns=ct.get_feature_names_out(), index=X_train.index)  # fit scaler to train data
        X_t = None
        if X_test is not None:
            X_t = pd.DataFrame(ct.transform(X_test), columns=ct.get_feature_names_out(), index=X_test.index)  # apply scaling to test data, too
        return X, X_t
    
    def scale_target(self, y_train, y_test):
        if y_train.name.startswith('BinCAT_') or y_train.name.startswith('CAT_'):  # categorical target: no scaling
            return y_train, y_test
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y_train.values.reshape(-1,1))
        y = pd.Series(y_scaled.flatten(), name=y_train.name, index=y_train.index)
        y_t = None
        if y_test is not None:
            y_scaled = scaler.transform(y_test.values.reshape(-1,1))
            y_t = pd.Series(y_scaled.flatten(), name=y_test.name, index=y_test.index)
        return y, y_t

def process_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--embH5', action="store", default="", help="Path of the raw emb.h5 file containining the results of the model (can be only just the folder containing emb.h5)")
    parser.add_argument('--embH5_heldout', action="store", default="", help="Path of the raw emb.h5 file containining the held-out results of the model (can be only just the folder containing emb.h5)")
    
    parser.add_argument('--res_path', action="store", default="", help="Fully-qualified path of the tsv file containining the results of the model (e.g., processed_raw.tsv)")
    parser.add_argument('--res_heldout_path', action="store", default="", help="(Optional) fully-qualified path of the tsv file containining the results of the model (e.g., processed_raw.tsv) for the held-out test subjects - Only considered with --res_path.")
    parser.add_argument('--merged_path', action="store", default="", help="Fully-qualified path of the tsv file containining the results of a merge (e.g., merged_latents_raw.tsv).")
    parser.add_argument('--merged_heldout_path', action="store", default="", help="(Optional) fully-qualified path of the tsv file containining the results of a merge (e.g., merged_latents_raw.tsv) for the held-out test subjects - Only considered with --merged_path.")
    parser.add_argument('--pred_path', action="store", default="", help="Fully-qualified path of the tsv file containining the predictors (e.g., clinicaldata_cardiacfunc.tsv).")
    parser.add_argument('--pred_heldout_path', action="store", default="", help="(Optional) fully-qualified path of the tsv file containining the predictors (e.g., clinicaldata_cardiacfunc.tsv) for the held-out test subjects - Only considered with --pred_path.")
    
    parser.add_argument('--complex_model', action=argparse.BooleanOptionalAction, default=False, help="Whether or not the model that is being evaluated is complex-valued. If yes, evals will be performed for the provided complex modes")
    # parser.add_argument('--complex_modes', action="store", default="real,imag,mag,phase,cartesian,polar,dualcoords", help="Coma-separated list of complex modes to evaluate [If complex_model is True]") 
    parser.add_argument('--complex_modes', action="store", default="cartesian,polar,dualcoords", help="Coma-separated list of complex modes to evaluate [If complex_model is True]") 

    parser.add_argument('--cov_path', action="store", default="", help="Fully-qualified path of the tsv file containining the basic covariates.")
    parser.add_argument("--cov_cols", action="store", default="BMI,Sex,Age,MRI_Date,MRI_Centre", help="Comma-separated list of the (correction) covariates for the association analysis. Leave it blank if association analysis is not desired.")
    parser.add_argument("--cov_bincat", action="store", default="Sex", help="Comma-separated list of the (correction) binary covariates for the association analysis. Leave it blank if association analysis is not desired.")
    parser.add_argument("--cov_cat", action="store", default="MRI_Date,MRI_Centre", help="Comma-separated list of the (correction) categorical covariates for the association analysis. Leave it blank if association analysis is not desired.")
        
    parser.add_argument('--model_tag', action="store", default="prova", help="Tag to identify the model.")

    parser.add_argument('--out_path', action="store", default="", help="Path to store the results. Make it blank if if it's desired to not store the results, rather just print them on the console.")
    
    parser.add_argument('--instance_merge', action=argparse.BooleanOptionalAction, default=True, help="Whether to merge considering the acquisition instance (Only if embH5 is provided).")
    parser.add_argument('--path_tsvs', action="store", default="../clinicaldata/abdominalMRI", help="Location of the tsv files, to be used for analyses. Leave it blank if fully-qualified paths are supplied in --tsv_files.")
    parser.add_argument('--tsv_files', action="store", default="Abdominal_composition_82779_MD_01_08_2024_16_54_05;Abdominal_organ_composition_82779_MD_01_08_2024_16_56_07", help="Coma-separated list of tsv files - to be joined and considered. To consider in one analysis, seperate by coma, to be considered in seperate ones, seperate by semicolon.")
    
    parser.add_argument('--predictors2ignore', action="store", default="", help="Coma-separated list of predictors (latents) to ignore.")  #CAT_Ethnicity,Standard_PRS_for_cardiovascular_disease_(CVD).0.0
    parser.add_argument('--attributes2ignore', action="store", default="CAT_Ethnicity", help="Coma-separated list of attributes to ignore [If they are covars, add _COV suffix]") #Reasons for the default vals: Birth_Month makes no sense, Smoking_Imaging is considered, Ethnicity is mostly white, BMI_Imaging is considered, MRIvisit shouldn't have any impact (but we should check it out in the future)
    
    parser.add_argument("--path_norm", action="store", default="", help="Fully-qualified file path of the normalising attribute (i.e., continuous targets are divided by this attribute) for conditioning. Leave it blank if target normalisation is not desired.")
    parser.add_argument("--attribute_norm", action="store", default="", help="Name of the normalising attribute to retrieve (i.e., continuous targets are divided by this attribute) for conditioning. Leave it blank if target normalisation is not desired.")

    parser.add_argument('--add_feat_path', action="store", default="", help="Fully-qualified path of the tsv file containining the additional features used for prediction. Leave it blank if not desired.")
    parser.add_argument('--add_feat2ignore', action="store", default="CAT_Ethnicity", help="Coma-separated list of additional features to ignore.")  #Standard_PRS_for_cardiovascular_disease_(CVD).0.0
    
    parser.add_argument('--use_feature_scaling', action=argparse.BooleanOptionalAction, default=False, help="Whether to use feature scaling or not.")
    parser.add_argument('--use_target_scaling', action=argparse.BooleanOptionalAction, default=True, help="Whether to use target scaling or not (continuous targets only).")
    parser.add_argument('--remove_with_VIF', action=argparse.BooleanOptionalAction, default=False, help="Whether to remove VIFs.")

    parser.add_argument('--run_also_trainonly', action=argparse.BooleanOptionalAction, default=False, help="In case held-out test set is not provided, whether to run train-only models (only to get coefficients from the whole DS).")
    
    parser.add_argument('--suppress_warnings', action=argparse.BooleanOptionalAction, default=True, help="Whether to suppress the known warnings.")
    
    parser.add_argument('--n_jobs', action="store", type=int, default=10, help="Number of jobs to use for parallelisation.")
    parser.add_argument('--max_iter_CV', action="store", type=int, default=1000, help="Maximum number of iteration for the cross-validation tasks.")
    parser.add_argument('--max_iter', action="store", type=int, default=100, help="Maximum number of iteration for the non-cross-validation, actual model learning tasks.")

    parser.add_argument("--l1_penalty", action="store", type=float, default=0.05, help="Custom scikit-learn penalization hyperparameter for L1 loss, besides L1_CV. Set it <=0 if you only want to use L1_CV.")

    parser.add_argument('--n_folds', action="store", type=int, default=5, help="Number of folds for k-fold model assessment.")

    args, unknown_args = parser.parse_known_args()

    args.predictors2ignore = args.predictors2ignore.split(',')
    args.attributes2ignore = args.attributes2ignore.split(',')
    args.add_feat2ignore = args.add_feat2ignore.split(',')

    return args, unknown_args

def main():
    args, unknown_args = process_arguments()
    
    if args.suppress_warnings:
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    orig_model_tag = args.model_tag
    args.out_path = os.path.join(args.out_path, args.model_tag)

    if args.complex_model:
        args.complex_modes = args.complex_modes.split(',')
    else:
        args.complex_modes = ['notcomplex']
    
    tsv_file_combos = args.tsv_files.split(';')
    for tsv_file_combo in tsv_file_combos:
        args.tsv_files = tsv_file_combo

        for complex_mode in args.complex_modes:
            if complex_mode != 'notcomplex':
                args.complex_mode = complex_mode
                args.model_tag = f"{orig_model_tag}_{complex_mode}"

            analyser = MLAnalyses(vars(args))

            parallel_process_attributes(analyser, args, num_threads=args.n_jobs)
            
            analyser.save_results(flush_results=True)

if __name__ == "__main__":
    main()