# Enabling Intel Extension for Scikit-learn (comment it out if not required/working)
from sklearnex import patch_sklearn
patch_sklearn()

import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, LassoCV, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

res_collect = defaultdict(dict)

## VIF analysis and removal
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

def remove_high_vif_features(X, threshold=10):
    while True:
        vif = calculate_vif(X)
        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            max_vif_feature = vif.loc[vif["VIF"] == max_vif, "feature"].values[0]
            X = X.drop(columns=[max_vif_feature])
            print(f"Removed feature: {max_vif_feature} with VIF: {max_vif}")
        else:
            break

    return X

####### Eval functions
def eval_linear(model, features, X_train, y_train, X_test=None, y_test=None, exp_type=""):
    if X_test is not None:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        if bool(exp_type):
            res_collect[exp_type]['MSE_TestSet'] = mse
            res_collect[exp_type]['R-squared_TestSet'] = r2
        else:
            print(f'Mean Squared Error (Test Set): {mse}')
            print(f'R-squared (Test Set): {r2}')

    coefficients = pd.DataFrame(model.coef_, index=features, columns=['Coefficient'])
    
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

    if bool(exp_type):
        res_collect[exp_type]['Coefficients'] = coefficients
    else:
        print("Coefficients with t-values and p-values:")
        print(coefficients)

    significance_level = 0.05
    significant_coefficients = coefficients[coefficients['p-value'] < significance_level]

    if bool(exp_type):
        res_collect[exp_type]['SignificantCoefficients'] = significant_coefficients
    else:
        print("\nSignificant coefficients:")
        print(significant_coefficients)


def eval_logistic(model, features, X_train, X_test=None, y_test=None, exp_type=""):
    if X_test is not None:
        y_pred = model.predict(X_test)
        if bool(exp_type):
            df_rs = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
            res_collect[exp_type]['ClassifRprt_TestSet'] = df_rs
        else:
            print("Classification report:")
            print(classification_report(y_test, y_pred))

    coefficients = pd.DataFrame(model.coef_.T, index=features, columns="Cofficient_class_"+model.classes_)

    for cls in"Cofficient_class_"+model.classes_:
        coefficients[cls.replace("Cofficient", "OddsRatio")] = np.exp(coefficients[cls]) # Calculate odds ratios

        # Calculate the standard errors
        X_std = X_train.std()
        standard_errors = X_std * np.sqrt(np.diag(np.linalg.inv(np.dot(X_train.T, X_train))))

        # Calculate the confidence intervals
        z = stats.norm.ppf(1 - 0.05 / 2)
        coefficients[cls.replace("Cofficient", "LowerCI")] = np.exp(coefficients[cls] - z * standard_errors)
        coefficients[cls.replace("Cofficient", "UpperCI")] = np.exp(coefficients[cls] + z * standard_errors)

    if bool(exp_type):
        res_collect[exp_type]['Coefficients'] = coefficients
    else:
        print("Odds Ratios and Confidence Intervals:")
        print(coefficients)

    for cls in"Cofficient_class_"+model.classes_:
        # Identify significant coefficients based on confidence intervals
        significant_coefficients = coefficients[(coefficients[cls.replace("Cofficient", "LowerCI")] > 1) | (coefficients[cls.replace("Cofficient", "UpperCI")] < 1)]
        
        if bool(exp_type):
            res_collect[exp_type][f'SignificantCoefficients_{cls}'] = significant_coefficients
        else:
            print(f"\nSignificant coefficients for {cls}:")
            print(significant_coefficients)

            
########################

##### Single model functions

def single_linear_regression(X_train, X_test, y_train, y_test, latent_factors, exp_type):
    model = LinearRegression(n_jobs=10)
    model.fit(X_train, y_train)

    eval_linear(model=model, features=latent_factors, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, exp_type=exp_type)

def single_logistic_regression(X_train, X_test, y_train, y_test, latent_factors, is_binary, exp_type):
    model = LogisticRegression(max_iter=1000, solver="saga", multi_class='ovr' if is_binary else 'multinomial', n_jobs=10)
    model.fit(X_train, y_train)
    
    eval_logistic(model=model, features=latent_factors, X_train=X_train, X_test=X_test, y_test=y_test, exp_type=exp_type)

########################

#RFE functions

def RFE_linaer_regression(X_train, X_test, y_train, y_test, latent_factors, exp_type):
    for n_feat in range(2, len(latent_factors)):
        model = LinearRegression(n_jobs=10)

        # Perform RFE to find the most important features
        selector = RFE(model, n_features_to_select=n_feat, step=1) 
        selector = selector.fit(X_train, y_train)

        # Display the selected features
        selected_features = [latent_factors[i] for i, mask in enumerate(selector.support_) if mask]
        if bool(exp_type):
            res_collect[f"{exp_type}_nfeat{n_feat}"][f'SelectedFeatures'] = selected_features
        else:
            print("Selected features:", selected_features)

        # Train the model with the selected features
        X_selected = X_train[selected_features]
        model.fit(X_selected, y_train)
        
        eval_linear(model=model, features=selected_features, X_train=X_selected, y_train=y_train, X_test=X_test[selected_features] if X_test is not None else None, y_test=y_test, exp_type=f"{exp_type}_nfeat{n_feat}" if bool(exp_type) else "")


def RFE_logistic_regression(X_train, X_test, y_train, y_test, latent_factors, is_binary, exp_type):
    for n_feat in range(2, len(latent_factors)):
        model = LogisticRegression(max_iter=1000, solver="saga", multi_class='ovr' if is_binary else 'multinomial', n_jobs=10)

        # Perform RFE to find the most important features
        selector = RFE(model, n_features_to_select=n_feat, step=1) 
        selector = selector.fit(X_train, y_train)

        # Display the selected features
        selected_features = [latent_factors[i] for i, mask in enumerate(selector.support_) if mask]
        if bool(exp_type):
            res_collect[f"{exp_type}_nfeat{n_feat}"][f'SelectedFeatures'] = selected_features
        else:
            print("Selected features:", selected_features)

        # Train the model with the selected features
        X_selected = X_train[selected_features]
        model.fit(X_selected, y_train)

        eval_logistic(model=model, features=selected_features, X_train=X_selected, X_test=X_test[selected_features] if X_test is not None else None, y_test=y_test, exp_type=f"{exp_type}_nfeat{n_feat}" if bool(exp_type) else "")

########################

#Lasso functions
def lasso(X_train, X_test, y_train, y_test, latent_factors, exp_type):
    lasso_cv = LassoCV(alphas=None, cv=5, max_iter=10000, random_state=1701, n_jobs=10)
    lasso_cv.fit(X_train, y_train)
    optimal_alpha = lasso_cv.alpha_
    if bool(exp_type):
        res_collect[exp_type]['OptimalAlpha'] = optimal_alpha
    else:
        print(f"Optimal alpha value: {optimal_alpha}")

    model = Lasso(alpha=optimal_alpha, max_iter=10000, random_state=1701)
    model.fit(X_train, y_train)

    if X_test is not None:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        if bool(exp_type):
            res_collect[exp_type]['MSE_TestSet'] = mse
            res_collect[exp_type]['R-squared_TestSet'] = r2
        else:
            print(f'Mean Squared Error (Test Set): {mse}')
            print(f'R-squared (Test Set): {r2}')

    coefficients = pd.DataFrame(model.coef_, index=latent_factors, columns=['Coefficient'])

    if bool(exp_type):
        res_collect[exp_type]['Coefficients'] = coefficients
    else:
        print("Coefficients:")
        print(coefficients)

    significant_coefficients = coefficients[coefficients['Coefficient'] != 0]

    if bool(exp_type):
        res_collect[exp_type]['SignificantCoefficients'] = significant_coefficients
    else:
        print("\nSignificant coefficients:")
        print(significant_coefficients)

def lasso_categorical(X_train, X_test, y_train, y_test, latent_factors, is_binary, exp_type):
    # LogisticRegressionCV model with L1 regularization -  essentially the logistic regression equivalent of Lasso.
    logreg_cv = LogisticRegressionCV(Cs=10, cv=5, penalty='l1', solver='saga', max_iter=10000, random_state=1701, multi_class='ovr' if is_binary else 'multinomial', n_jobs=10)
    logreg_cv.fit(X_train, y_train)
    optimal_C = logreg_cv.C_[0]
    if bool(exp_type):
        res_collect[exp_type]['OptimalC'] = optimal_C
    else:
        print(f"Optimal C value: {optimal_C}")

    model = LogisticRegression(C=optimal_C, penalty='l1', solver='saga', max_iter=10000, random_state=1701, multi_class='ovr' if is_binary else 'multinomial', n_jobs=10)
    model.fit(X_train, y_train)

    eval_logistic(model=model, features=latent_factors, X_train=X_train, X_test=X_test, y_test=y_test, exp_type=exp_type)

###############
path_csv = "/project/ukbblatent/processed_latents/time2ch_4ChTrans_FactorVAE_latent64/dataframes/cardiac_baseline_emb.csv"
out_path = "/project/ukbblatent/processed_latents/time2ch_4ChTrans_FactorVAE_latent64/MLAnalysis"
n_latent_factors = 64
attributes2ignore = ["Birth_Month", "Smoking", "Ethnicity", "BMI", "MRIvisit"] #Reasons: Birth_Month makes no sense, Smoking_Imaging is considered, Ethnicity is mostly white, BMI_Imaging is considered, MRIvisit shouldn't have any impact (but we should check it out in the future)
binary_attributes = ["Gender", "Smoking_Imaging"]
use_feature_scaling = True
remove_with_VIF = True
###############
print("10 jobs")

df = pd.read_csv(path_csv, index_col=0)

attributes = [col for col in df.columns if not col.startswith('Z') and col not in attributes2ignore]

latent_factors = [f"Z{i}" for i in range(n_latent_factors)]
X = df[latent_factors]

if use_feature_scaling:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=latent_factors, index=df.index)

if remove_with_VIF:
    X = remove_high_vif_features(X, threshold=10)

for target_attribute in tqdm(attributes): 
    y = df[target_attribute]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1701)

    if "CAT" in target_attribute or target_attribute in binary_attributes: #It's categorical, run Logistic Regression
        single_logistic_regression(X_train, X_test, y_train, y_test, latent_factors, is_binary=target_attribute in binary_attributes, exp_type=f"{target_attribute}_LogRegrs_wTest")
        RFE_logistic_regression(X_train, X_test, y_train, y_test, latent_factors, is_binary=target_attribute in binary_attributes, exp_type=f"{target_attribute}_RFELogRegrs_wTest")
        lasso_categorical(X_train, X_test, y_train, y_test, latent_factors, is_binary=target_attribute in binary_attributes, exp_type=f"{target_attribute}_LassoCat_wTest")

        single_logistic_regression(X, None, y, None, latent_factors, is_binary=target_attribute in binary_attributes, exp_type=f"{target_attribute}_LogRegrs_woTest")
        RFE_logistic_regression(X, None, y, None, latent_factors, is_binary=target_attribute in binary_attributes, exp_type=f"{target_attribute}_RFELogRegrs_woTest")
        lasso_categorical(X, None, y, None, latent_factors, is_binary=target_attribute in binary_attributes, exp_type=f"{target_attribute}_LassoCat_woTest")
    else:
        single_linear_regression(X_train, X_test, y_train, y_test, latent_factors, exp_type=f"{target_attribute}_LinRegrs_wTest")
        RFE_linaer_regression(X_train, X_test, y_train, y_test, latent_factors, exp_type=f"{target_attribute}_RFELinRegrs_wTest")
        lasso(X_train, X_test, y_train, y_test, latent_factors, exp_type=f"{target_attribute}_Lasso_wTest")

        single_linear_regression(X, None, y, None, latent_factors, exp_type=f"{target_attribute}_LinRegrs_woTest")
        RFE_linaer_regression(X, None, y, None, latent_factors, exp_type=f"{target_attribute}_RFELinRegrs_woTest")
        lasso(X, None, y, None, latent_factors, exp_type=f"{target_attribute}_Lasso_woTest")


if bool(out_path):
    os.makedirs(out_path, exist_ok=True)
    with open(f"{out_path}/{os.path.basename(path_csv).split('.')[0]}_results_intel_10jobs.pickle", "wb") as f:
        pickle.dump(res_collect, f)