# %% [markdown]
# # Datasets for latent predictability of (binary) endpoints/diseases

# %%
import os
import sys
import argparse
import pandas as pd

# sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/soumick.chatterjee/Codes/GitLab/tricorder")

from analyses.disease_mappings import *

# %%
def select_values(row, base_cols, suffixes):
    for suffix in suffixes:
        if all(not pd.isna(row[f"{col}.{suffix}.0"]) for col in base_cols):
            return [row[f"{col}.{suffix}.0"] for col in base_cols] + [suffix]
    return [pd.NA] * len(base_cols)
    
def get_top_diseases(phenos, no_subjects=200):    
    # Get diseases with at least no_subjects subjects within the year range
    disease_counts = phenos.reset_index().groupby('summary')['eid'].nunique()
    incident_diseases = disease_counts[disease_counts >= no_subjects].index.tolist()
    
    # Print the number of diseases that have more than no_subjects subjects
    print(f"Number of diseases with more than {no_subjects} subjects: {len(incident_diseases)}")
    return incident_diseases

def get_corresponding(source_df, age, sex, set_avoid):
    age_diff = 0  # Start with no age difference
    max_age_diff = max(abs(source_df["Age"].min() - age), abs(source_df["Age"].max() - age))  # Maximum possible age difference
    set_avoid = set(set_avoid)  # Ensure set_avoid is a set for efficient lookups

    while age_diff <= max_age_diff:
        # Check for subjects with the current age or age +/- age_diff
        for delta in (0, age_diff, -age_diff if age_diff > 0 else 0):
            current_age = age + delta
            corresp = set(source_df[(source_df["Age"] == current_age) & (source_df["Sex"] == sex)].index)
            corresp.difference_update(set_avoid)

            if corresp:  # If there's at least one matching subject
                return corresp.pop()

        age_diff += 1  # Increase the age difference for the next iteration

    return None

# %%

def getARGSParser():
    parser = argparse.ArgumentParser(description='MultiPRS Script')
    parser.add_argument('--in_path', type=str, help='Path to PRS results root directory', default="../clinicaldata/merge_SR_HI_GP_v4_allUKB_&_HEALTHY.csv")
    parser.add_argument('--covar_path', type=str, help='Prefix before the pheno name in the RDS file name', default="../PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/covars/nonDisc_caucasian_king0p0625_V0.tsv")
    parser.add_argument('--raw_baseline_path', type=str, help='Prefix before the pheno name in the RDS file name', default="../clinicaldata/v1.1.0_seventh_basket/baseline_MD_27_10_2023_13_10_05.tsv")
    parser.add_argument('--info_assessment_centre_path', type=str, help='Prefix before the pheno name in the RDS file name', default="../clinicaldata/v1.1.0_seventh_basket/assessmentCentre_82779_MD_13_06_2024_12_18_59.tsv")
    parser.add_argument('--subIDs_path', type=str, help='Suffix after the pheno name in the RDS file name', default="../PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/2_king_cutoff_0p0625_nonDisc_cond_plus_plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.king.cutoff.in.id")
    
    parser.add_argument('--out_path', type=str, help='Suffix after the pheno name in the RDS file name', default="../clinicaldata/binary_disease_cohorts/F20208v3_nonDiscov/caucasian_king0p0625_grouped/newcovsets/V0")
    parser.add_argument('--mode', type=int, help='Mode 0: All, 1: 5y prognosis, 2: 10y prognosis, 3: Ny prognosis, -1: Ny diagnosis', default=0)

    parser.add_argument('--healthy', type=str, help='Tag for healthy subjects', default="healthy")
    parser.add_argument('--target', type=str, help='Name of target column', default='BinCAT_Disease')
    
    parser.add_argument('--cutoff_date', type=str, help='To filter abnormal dates, a cutoff date must be used. Any entry in the future wrt this date will be removed.', default='2023-10-31')
    parser.add_argument('--min_days_prognosis', type=int, help='Lower-bound for prognosis.', default=0)

    parser.add_argument('--is_caucasian_only', action=argparse.BooleanOptionalAction, default=True, help='Run multiPRS models')
    parser.add_argument('--use_mappings', action=argparse.BooleanOptionalAction, default=True, help='Run multi normalised PRS models')

    parser.add_argument('--binarise_smoking', action=argparse.BooleanOptionalAction, default=False, help='Run multi normalised PRS models')

    return parser

parser = getARGSParser()
args = parser.parse_args()

mappings = cardiac_mappings
healthy = args.healthy

# %% [markdown]
# ## Load subjects with latent factors (for merging)

# %% [markdown]
# ## Load data

# %%
df = pd.read_csv(args.in_path, low_memory=False).set_index("eid")

if bool(args.subIDs_path):
    with open(args.subIDs_path, "r") as f:
        subIDs = f.readlines()
    subIDs = [int(s.split("\t")[0]) for s in subIDs[1:]]
    df = df[df.index.isin(subIDs)]

covar = pd.read_table(args.covar_path).set_index("IID")
if args.is_caucasian_only:
    if "Ethnic_gen" in covar.columns:
        covar = covar[covar.Ethnic_gen == 1]
    else:
        print("is_caucasian_only is set to True, but Ethnic_gen column is not present in the covariate file. Skipping Caucasian-only filtering.")

covar_reduced = covar.drop(columns=df.columns.intersection(covar.columns)) #remove the columns that are also present in df, to avoid _x and _y naming.
df = df.merge(covar_reduced, left_index=True, right_index=True) 

if bool(args.raw_baseline_path):
    raw_baseline = pd.read_table(args.raw_baseline_path).set_index("f.eid")
    raw_baseline = raw_baseline[[c for c in raw_baseline.columns if "f.20116." in c or "f.21001" in c]]
    assessment_centre = pd.read_table(args.info_assessment_centre_path).set_index("f.eid")
    assessment_centre = assessment_centre[[c for c in assessment_centre.columns if "f.53." in c]]
    df_unprocessed = df.merge(raw_baseline, left_index=True, right_index=True).merge(assessment_centre, left_index=True, right_index=True) 
    df_unprocessed.index.names = ['eid']
    df_unprocessed[['f.20116', 'f.21001', 'f.53', 'selected_instance']] = df_unprocessed.apply(lambda row: pd.Series(select_values(row, base_cols=['f.20116', 'f.21001', 'f.53'], suffixes=['0', '1', '2', '3'])), axis=1)
    df_unprocessed = df_unprocessed.drop([c for c in df_unprocessed.columns if "f.20116." in c or "f.21001." in c or "f.53." in c], axis=1) #we will remove now the instance-wise columns
    df_unprocessed['date'] = pd.to_datetime(df_unprocessed['date'], format='%Y-%m-%d', errors='coerce')
    df_unprocessed['f.53'] = pd.to_datetime(df_unprocessed['f.53'], format='%Y-%m-%d', errors='coerce')

    #remove subjects with "future" diseases
    df_unprocessed = df_unprocessed[((df_unprocessed['f.53'].isna()) | (df_unprocessed['f.53'] <= pd.to_datetime(args.cutoff_date))) & ((df_unprocessed['date'].isna()) | (df_unprocessed['date'] <= pd.to_datetime(args.cutoff_date)))]

    df_unprocessed['DiseaseAfter'] = (df_unprocessed['date'] - df_unprocessed['f.53']).dt.days
    del df_unprocessed['CAT_Smoking'], df_unprocessed['BMI']
    df_unprocessed.rename(columns={'f.20116': 'CAT_Smoking', 'f.21001': 'BMI', 'f.53': 'BaselineDate'}, inplace=True)

    df_unprocessed = df_unprocessed[df_unprocessed['CAT_Smoking'] != -3].copy()

    if args.binarise_smoking:
        df_unprocessed['CAT_Smoking'] = df_unprocessed['CAT_Smoking'].replace(2.0, 1.0)

    df_unprocessed = df_unprocessed.sort_values(by='date', na_position='last')
    df_unprocessed = df_unprocessed.reset_index()
    df_unprocessed = df_unprocessed[~df_unprocessed.duplicated(subset=['summary', 'eid'], keep='first')].sort_values(by='eid').set_index('eid')

    if args.mode in [1, 2, 3]:
        df_after = df_unprocessed[(df_unprocessed['date'].isna()) | (df_unprocessed['DiseaseAfter'] > args.min_days_prognosis)]
        
    match args.mode:
        case 0:
            df = df_unprocessed.copy()
        case 1:
            df = df_after[(df_after['date'].isna()) | (df_after['DiseaseAfter'] <= 5*365)].copy()
        case 2:
            df = df_after[(df_after['date'].isna()) | (df_after['DiseaseAfter'] <= 10*365)].copy()
        case 3:
            df = df_after.copy()
        case -1:
            df = df_unprocessed[(df_unprocessed['date'].isna()) | (df_unprocessed['DiseaseAfter'] <= 0)].copy()
        case _:
            sys.exit("Invalid mode. Exiting.")

elif args.mode != 0 or args.binarse_smoking:
    sys.exit("Mode is not 0 or smoking is being binarised, but raw_baseline_path is not provided. Exiting.")

df.info()
df.head()

# %%
if args.use_mappings:
    print(f"Currently, there are {df.summary.nunique()-1} diseases, and args.use_mappings has been set to True. So, grouped diseaes will be added now!")
    df = add_grouped_diseases(df, mappings, healthy)
    print(f"After adding grouped diseases (and removing the ones not mentioned in the mappings), we finally have: {df.summary.nunique()-1} diseases.")


# %%
df

# %%
diseases = df["summary"].unique()  # possible diseases
print(diseases)

# %% [markdown]
# ## Define disease-specific dataframes

# %%
disease_df = {}  # dictionary for disease-specific dataframes

for disease in diseases:
    disease_df[disease] = df[df["summary"] == disease].copy()
    print(disease, ':', disease_df[disease].index.nunique(), 'subjects')

# %%
# Example
example = "atrial fibrillation/flutter"
disease_df[example].info()
disease_df[example].head()
print(f"#subjects: {disease_df[example].index.nunique()}")

# %% [markdown]
# ## Integrate disease-specific dataframes with healthy controls

# %% [markdown]
# We will insert healthy controls to have 50/50 % for classification.
# Healthy controls will be chosen with same age and sex of unhealthy subjects. If no match, then the corresponding control will have the same sex and closest older age.

# %% [markdown]
# ### Integrate

# %%
source = disease_df[healthy]  # source dataframe to match corresponding

disease_control_df = {}
for disease in disease_df:  # iterate over dataframes
    if disease != healthy:  # only over disease dataframes (no healthy)
        print('\n\n---------------', disease)
        curr_df = disease_df[disease]  # current disease-specific dataframe
        subIDs = curr_df.index.unique()
        sub_integrate = set()  # set of (corresponding) subjects to be integrated
        print(len(subIDs), 'unhealthy subjects')

        for sub in subIDs:  # iterate over subjects, to get corresponding for each of them
            corresponding = get_corresponding(  # get corresponding of current subject
                                                source_df=source,
                                                age=curr_df.loc[sub, "Age"].iloc[0] if isinstance(curr_df.loc[sub, "Age"], pd.Series) else curr_df.loc[sub, "Age"],
                                                sex=curr_df.loc[sub, "Sex"].iloc[0] if isinstance(curr_df.loc[sub, "Sex"], pd.Series) else curr_df.loc[sub, "Sex"],
                                                set_avoid=sub_integrate
                                            )  
            if corresponding is not None:
                sub_integrate.add(corresponding)  # add the corresponding subject to the set of corresponding
            else:
                print(f"{disease}: No corresponding subject found for {sub}")

        # Add corresponding subjects to the current disease-specific dataframe
        print(len(sub_integrate), healthy, 'subjects found')
        sub_integrate = source.loc[list(sub_integrate), :].copy()  # get full information from corresponding subjects
        disease_control_df[disease] = pd.concat([curr_df, sub_integrate])  # integrate (concatenate)

# %%
# Example
example = "atrial fibrillation/flutter"
disease_control_df[example].info()
disease_control_df[example].head()
print(f"#subjects: {disease_control_df[example].index.nunique()}")

# %%
# remove duplicates
for disease in disease_control_df.keys():
    print('\n\n-----', disease)    
    disease_control_df[disease] = disease_control_df[disease][~disease_control_df[disease].index.duplicated(keep='first')] #remove duplicates
    print(disease_control_df[disease]["summary"].value_counts(dropna=False))

# %% [markdown]
# ## Final dataframes

# %% [markdown]
# Final dataframes should contain a binary column: 1 for disease, 0 otherwise

# %%
# disease_args.target column is binary for the presence of the disease
for disease in disease_control_df.keys():
    curr_df = disease_control_df[disease]
    curr_df[args.target] = curr_df["summary"].copy().map(lambda x: 0 if x==healthy else 1)

# %%
# Final check
for disease in disease_control_df.keys():
    print('\n\n-----', disease)
    print(disease_control_df[disease][args.target].value_counts(dropna=False))

# %%
# Save
os.makedirs(args.out_path, exist_ok=True)
for disease in disease_control_df.keys():
    disDF = disease_control_df[disease].reset_index()
    try:
        disDF['FID'] = disDF['eid']
        disDF['IID'] = disDF['eid']
    except:
        disDF['FID'] = disDF['index']
        disDF['IID'] = disDF['index']
        disDF['eid'] = disDF['index']
        del disDF['index']
    disease = disease.replace(" ", "-").replace("/", "_").replace("'", "")
    disDF.to_csv(f"{args.out_path}/{disease}.csv", index=False)
    print(f"{disease}: {disDF.shape[0]//2}")

