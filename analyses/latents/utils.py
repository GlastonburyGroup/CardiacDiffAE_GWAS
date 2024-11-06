import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.figure_factory as ff
import plotly.io as pio
import numpy as np
from dateutil.parser import parse
from sklearn.covariance import EllipticEnvelope

## Create merged DFs
def compute_age(row, col_year_of_birth="Year_of_birth", col_month_of_birth="CAT_Month_of_birth"):
    diff = parse(row.date).year - int(row[col_year_of_birth])
    if (parse(row.date).month < int(row[col_month_of_birth])) and (abs(parse(row.date).month - int(row[col_month_of_birth])) >= 6):
        diff -=1
    return diff

## processing dataframes from UKBB Puller

import re

def create_dict(df, custom_dict={}, default_order=[0,1,2,3,4,5]):
    column_dict = {} if len(custom_dict) == 0 or not bool(custom_dict) else custom_dict.copy()
    categories = pd.unique(list(map(lambda x: re.sub('[0-9]\.[0-9]$', '', x), df.columns)))

    for cat in categories:

        """if re.match("^f\.[0-9]+\.[0-9]+\.", cat):
            print(f'malformed column: {cat}')
            continue"""
        if np.isin(cat, list(custom_dict.keys()) ):
            column_dict[cat] = custom_dict[cat]
        else:
            column_dict[cat] = default_order

    return column_dict            

def clean_one_column(df, select_order=[]):
    start = 0

    for i in select_order:
        if i <= (df.shape[1]-1):
            break
        start+=1

    base_val = df.iloc[:,select_order[start]]
    for i in select_order[start:]:
        if i > (df.shape[1]-1):
            continue

        base_val = base_val.fillna(df.iloc[:,i])

        if(base_val.isna().sum() == 0):
            return base_val
    return base_val

def clean_df(df, column_dict):  
    new_col = []

    for col in column_dict:
        filtered_columns = list(map(lambda x: x.startswith(col), df.columns))
        new_col.append(clean_one_column(df.loc[:,filtered_columns], column_dict[col]))

    return pd.DataFrame(np.array(new_col).T, columns=column_dict)

def get_attributes_ukbbpuller(file_path, indices=[], thresh_notna=-1, cols_nodrop=[], df=None, cols2drop=[], col_names={}, custom_dict={}, default_order=[2,3,1,0,4,5,6]):
    if df is None:
        df = pd.read_table(file_path).set_index('f.eid')
    df.index = df.index.astype(int)   

    if len(indices) > 0:
        df = df[df.index.isin(indices)]

    if len(default_order) > 0: 
        df = df.reset_index() 
        df = clean_df(df, create_dict(df, custom_dict=custom_dict, default_order=default_order))
        df = df.set_index('f.eid')
        df.index = df.index.astype(int)
    else:
        new_columns = {}
        for column in df.columns:
            for key, value in col_names.items():
                if column.startswith(key):
                    if key.endswith('.'):
                        new_column = column.replace(key, value+'.')
                    else:
                        new_column = column.replace(key, value)
                    new_columns[column] = new_column
                    break
        col_names = new_columns

    if len(cols2drop) > 0:
        df = df.drop(columns=cols2drop, axis=1)

    if len(cols_nodrop) > 0:
        df_drop = df.loc[:, ~df.columns.isin(cols_nodrop)]
        if thresh_notna != -1:
            df_drop = df_drop.dropna(axis=1, thresh=df_drop.shape[0]*thresh_notna)
        df = pd.concat([df_drop, df.loc[:, cols_nodrop]], axis=1)
    elif thresh_notna != -1:
        df = df.dropna(axis=1, thresh=df.shape[0]*thresh_notna)

    return df.rename(columns=col_names)
        

def remove_outliers(df, contamination=0.01, support_fraction=0.63):
    df = df.dropna() #cannot be applied to df with NaNs
    ee = EllipticEnvelope(contamination=contamination, support_fraction=support_fraction, random_state=1701)
    yhat = ee.fit_predict(df) 
    mask = yhat != -1
    return df[mask]

## ML Analyses support utils

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

def categorise_corrcoef_strength(corr_coeff):
    abs_corr_coeff = abs(corr_coeff)
    if 0.00 <= abs_corr_coeff < 0.20:
        return 'Very weak'
    elif 0.20 <= abs_corr_coeff < 0.40:
        return 'Weak'
    elif 0.40 <= abs_corr_coeff < 0.60:
        return 'Moderate'
    elif 0.60 <= abs_corr_coeff < 0.80:
        return 'Strong'
    else:  # 0.80 <= abs_corr_coeff <= 1.00
        return 'Very strong'
    
## Plotting utils

def PlotNSaveHeatmap(matrix, title, fig_size=1500, font_size=8, rootpath="", filename=""):
    matrix_rounded = matrix.round(2)

    fig = ff.create_annotated_heatmap(
        z=matrix_rounded.values,
        x=matrix_rounded.columns.tolist(),
        y=matrix_rounded.index.tolist(),
        annotation_text=matrix_rounded.values,
        colorscale='Viridis',
        hoverinfo='z',
        showscale=True
    )

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = font_size

    fig.update_layout(
        title=title,
        width=fig_size,
        height=fig_size,
        xaxis=dict(side='bottom'),
        margin=dict(t=100)
    )

    pio.write_html(fig, file=f'{rootpath}/{filename}.html', auto_open=False)