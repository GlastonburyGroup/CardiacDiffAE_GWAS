import numpy as np
import re

error_mapping = {
    #abdominal compositation
    "10P liver PDFF mean error indicator": ["10P Liver PDFF (proton density fat fraction)"],
    "ASAT error indicator": ["Abdominal subcutaneous adipose tissue volume (ASAT)"],
    "Anterior thigh error indicator (left)": ["Anterior thigh muscle fat infiltration (MFI) (left)", "Anterior thigh fat-free muscle volume (left)"],
    "Anterior thigh error indicator (right)": ["Anterior thigh muscle fat infiltration (MFI) (right)", "Anterior thigh fat-free muscle volume (right)"],
    "FR liver PDFF mean error indicator": ["FR liver PDFF mean"],
    "Posterior thigh error indicator (left)": ["Posterior thigh muscle fat infiltration (MFI) (left)", "Posterior thigh fat-free muscle volume (left)"],
    "Posterior thigh error indicator (right)": ["Posterior thigh muscle fat infiltration (MFI) (right)", "Posterior thigh fat-free muscle volume (right)"],
    "VAT error indicator": ["Visceral adipose tissue volume (VAT)"]
}

def correct_errors_raw(df):
    all_cols_error = []
    for (col_error, cols_item) in error_mapping.items():
        all_cols_error += list(df.filter(regex=f"^{re.escape(col_error.replace(' ', '_'))}*").columns) 
        for col_item in cols_item:
            for i in range(4):
                error_cols = df.filter(regex=f"^{re.escape(col_error.replace(' ', '_'))}\.{i}\.\d+$").columns
                item_cols = df.filter(regex=f"^{re.escape(col_item.replace(' ', '_'))}\.{i}\.\d+$").columns
                if len(error_cols) > 0 and len(item_cols) > 0:
                    error_condition = df[error_cols].notna().any(axis=1)
                    df.loc[error_condition, item_cols] = np.nan
    df = df.drop(columns=all_cols_error)
    return df.dropna(how='all')

def correct_errors_flat(df):
    for (col_error, cols_item) in error_mapping.items():
        for col_item in cols_item:
            df.loc[df[col_error.replace(" ", "_")].notna(), col_item.replace(" ", "_")] = np.nan
    df = df.drop(columns=list([k.replace(" ", "_") for k in error_mapping.keys()]))
    return df.dropna(how='all')