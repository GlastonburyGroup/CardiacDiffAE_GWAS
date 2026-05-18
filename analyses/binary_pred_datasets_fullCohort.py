import os
import pandas as pd
import numpy as np
import argparse
import pickle

from tqdm import tqdm

from icdmappings import Mapper

mapper = Mapper()

def select_values(row, base_cols, suffixes):
    for suffix in suffixes:
        if all(not pd.isna(row[f"{col}.{suffix}.0"]) for col in base_cols):
            return [row[f"{col}.{suffix}.0"] for col in base_cols] + [suffix]
    return [pd.NA] * len(base_cols)

def get_whole_cohort(cov_root="/group/glastonbury/GWAS/inputs/covariates/wholeCohort", cov_suffix="woSmoking"):

    imSubs  = {}
    
    df=pd.read_table(f"{cov_root}/cov_wholeCohort_{cov_suffix}.tsv")
    imSubs["wholeCohort"] = set(df.FID.to_list())

    df=pd.read_table(f"{cov_root}/cov_wholeCohort_{cov_suffix}_caucasian.tsv")
    imSubs["wholeCohort_caucasian"] = set(df.FID.to_list())

    return imSubs

def create_cohorts(cov_root, info_assessment_centre_path, imSubs, all3Phe, cutoff_date, cov_suffix="woSmoking"):
    #considering only the subjects when we have the same instance for all the modalities
    
    assessment_centre = pd.read_table(info_assessment_centre_path).set_index("f.eid")
    assessment_centre = assessment_centre[[c for c in assessment_centre.columns if "f.53." in c]]
    assessment_centre.index.names = ['eid']
    assessment_centre[['f.53', 'selected_instance']] = assessment_centre.apply(lambda row: pd.Series(select_values(row, base_cols=['f.53'], suffixes=['0', '1', '2', '3'])), axis=1)
    assessment_centre = assessment_centre.drop([c for c in assessment_centre.columns if "f.53." in c], axis=1) #we will remove now the instance-wise columns
    assessment_centre['f.53'] = pd.to_datetime(assessment_centre['f.53'], format='%Y-%m-%d', errors='coerce')

    #remove subjects with "future" diseases
    assessment_centre = assessment_centre[((assessment_centre['f.53'].isna()) | (assessment_centre['f.53'] <= pd.to_datetime(cutoff_date)))]

    assessment_centre.rename(columns={'f.53': 'BaselineDate'}, inplace=True)

    assessment_centre = assessment_centre.sort_values(by='BaselineDate', na_position='last')
    assessment_centre = assessment_centre.reset_index()

    cov_cohorts = {}
    dis_cohorts = {}
    disnodis_cohorts = {}
    for cohort in ["wholeCohort", "wholeCohort_caucasian"]:
        print(cohort)

        if cohort == "wholeCohort":
            cov = pd.read_csv(f"{cov_root}/cov_wholeCohort_{cov_suffix}.tsv", sep="\t")
        else:
            cov = pd.read_csv(f"{cov_root}/cov_wholeCohort_{cov_suffix}_caucasian.tsv", sep="\t")
        cov = cov.merge(assessment_centre, how="left", left_on="FID", right_on="eid").dropna()
            
        cov_cohorts[cohort] = cov[cov.FID.isin(imSubs[cohort])].copy()
        dis_cohorts[cohort] = cov_cohorts[cohort].merge(all3Phe[all3Phe.eid.isin(cov_cohorts[cohort].FID)], how="left", left_on="FID", right_on="eid").dropna(subset=["Phecode"])
        dis_cohorts[cohort]['Diag2Baseline_Year_Diff'] = np.ceil((dis_cohorts[cohort]['BaselineDate'] - dis_cohorts[cohort]['date']).dt.days / 365.25).astype(int)
        
        healthy = cov_cohorts[cohort][~cov_cohorts[cohort].FID.isin(dis_cohorts[cohort].FID)].copy()
        healthy["Phecode"] = "00_000.00"
        healthy["PhecodeLvl1"] = "00_000.0"
        healthy["PhecodeCh"] = "00_000"
        healthy["PhecodeString"] = "NoDisease"
        healthy["PhecodeCategory"] = "PseudoHealthy"
        healthy["Diag2Baseline_Year_Diff"] = 0
        disnodis_cohorts[cohort] = pd.concat([dis_cohorts[cohort], healthy])    

    return cov_cohorts, dis_cohorts, disnodis_cohorts

#original function
# def get_corresponding(source_df, age, sex, set_avoid):
#     age_diff = 0  # Start with no age difference
#     max_age_diff = max(abs(source_df["Age"].min() - age), abs(source_df["Age"].max() - age))  # Maximum possible age difference
#     set_avoid = set(set_avoid)  # Ensure set_avoid is a set for efficient lookups

#     while age_diff <= max_age_diff:
#         # Check for subjects with the current age or age +/- age_diff
#         for delta in (0, age_diff, -age_diff if age_diff > 0 else 0):
#             current_age = age + delta
#             corresp = set(source_df[(source_df["Age"] == current_age) & (source_df["Sex"] == sex)].index)
#             corresp.difference_update(set_avoid)

#             if corresp:  # If there's at least one matching subject
#                 return corresp.pop()

#         age_diff += 1  # Increase the age difference for the next iteration

#     return None

def get_corresponding(source_df, age, sex, set_avoid):
    set_avoid = set(set_avoid)  # Ensure set_avoid is a set for efficient lookups
    potential_matches = source_df[(source_df["Sex"] == sex) & (~source_df.index.isin(set_avoid))].copy()
    
    if potential_matches.empty:
        return None

    potential_matches["age_diff"] = (potential_matches["Age"] - age).abs()
    closest_match = potential_matches.loc[potential_matches["age_diff"].idxmin()]

    return closest_match.name if closest_match["age_diff"] <= max(abs(potential_matches["Age"].min() - age), abs(potential_matches["Age"].max() - age)) else None

def create_dis_cohorts(cohortDF, mode, code_col, min_subs=100):
    #mode: "all", "diag", "diag10y", "diag5y" "prog", "prog5y" "near", "near5y", "near2y"
    #code_col: "Phecode", "PhecodeCh", "PhecodeLvl1"
    cohortDF = cohortDF.rename(columns={"FID": "f.eid"}).drop(columns="IID")
    cohortDF = cohortDF.set_index(["f.eid"])
    unique_codes = cohortDF[code_col].unique()

    disDFs = {}
    for c in tqdm(unique_codes):
        if c.startswith("00_000"):
            continue

        disDF = cohortDF[cohortDF[code_col] == c].copy()
        if mode == "diag":
            disDF = disDF[disDF.Diag2Baseline_Year_Diff >= 0]
        elif mode == "diag10y":
            disDF = disDF[(disDF.Diag2Baseline_Year_Diff >= 0) & (disDF.Diag2Baseline_Year_Diff <= 10)]
        elif mode == "diag5y":
            disDF = disDF[(disDF.Diag2Baseline_Year_Diff >= 0) & (disDF.Diag2Baseline_Year_Diff <= 5)]
        elif mode == "prog":
            disDF = disDF[disDF.Diag2Baseline_Year_Diff < 0]
        elif mode == "prog5y":
            disDF = disDF[(disDF.Diag2Baseline_Year_Diff < 0) & (disDF.Diag2Baseline_Year_Diff >= -5)]
        elif mode == "near":
            disDF = disDF[(disDF.Diag2Baseline_Year_Diff <= 5) & (disDF.Diag2Baseline_Year_Diff >= -2)]
        elif mode == "near5y":
            disDF = disDF[(disDF.Diag2Baseline_Year_Diff <= 5) & (disDF.Diag2Baseline_Year_Diff >= -5)]
        elif mode == "near2y":
            disDF = disDF[(disDF.Diag2Baseline_Year_Diff <= 2) & (disDF.Diag2Baseline_Year_Diff >= -2)]
        disDF = disDF.iloc[
                    disDF.reset_index()  
                    .assign(abs_diff=lambda x: x['Diag2Baseline_Year_Diff'].abs())  
                    .groupby(['f.eid'])['abs_diff']  
                    .idxmin()   #we only need the occurence closest to the MRI
                ]
        subIDs = disDF.index.unique()
        if len(subIDs) < min_subs:
            print(f"{c}: Not enough subjects ({len(subIDs)})")
            continue
        disDF["BinCAT_Disease"] = 1

        nondisDF = cohortDF[cohortDF.PhecodeCh != disDF.PhecodeCh.iloc[0]].copy()
        nondisDF = nondisDF[~nondisDF.index.isin(subIDs)]
        nondisDF = nondisDF.iloc[
                    nondisDF.reset_index()  
                    .assign(abs_diff=lambda x: x['Diag2Baseline_Year_Diff'].abs())  
                    .groupby('f.eid')['abs_diff']  
                    .idxmin()  #we only need the occurence closest to the MRI, as we don't care about the disease
                ]
        sub_integrate = set()
        for sub in subIDs:
            corresponding = get_corresponding(  # get corresponding of current subject
                                                source_df=nondisDF,
                                                age=disDF.loc[sub, "Age"].iloc[0] if isinstance(disDF.loc[sub, "Age"], pd.Series) else disDF.loc[sub, "Age"],
                                                sex=disDF.loc[sub, "Sex"].iloc[0] if isinstance(disDF.loc[sub, "Sex"], pd.Series) else disDF.loc[sub, "Sex"],
                                                set_avoid=sub_integrate
                                            )  
            if corresponding is not None:
                sub_integrate.add(corresponding)  # add the corresponding subject to the set of corresponding
            else:
                print(f"{c}: No corresponding subject found for {sub}")
        nondisDF = nondisDF.loc[list(sub_integrate), :].copy()
        nondisDF["BinCAT_Disease"] = 0
        nondisDF.loc[nondisDF["PhecodeCh"] != "00_000", "Phecode"] = "00_000.10"
        nondisDF.loc[nondisDF["PhecodeCh"] != "00_000", "PhecodeLvl1"] = "00_000.1"
        nondisDF.loc[nondisDF["PhecodeCh"] != "00_000", "PhecodeString"] = "WoThisDisease"
        nondisDF["PhecodeCh"] = "00_000"
        nondisDF["PhecodeCategory"] = "PseudoHealthy"
        nondisDF["Diag2Baseline_Year_Diff"] = 0
        nondisDF[["date", "diag_type", "ICD", "ICDch", "Flag", "ICDString", "meaning", "code", "code_format", "year"]] = np.nan
        
        disDFs[c] = pd.concat([disDF, nondisDF])

    return disDFs
    
def process_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--processed_phecodes', action="store", default='/project/ukbblatent/clinicaldata/Atlas/mappedPheX/processed_GP_HI_SR_Phecode_ALL.csv') 
    parser.add_argument('--info_assessment_centre_path', type=str, help='Prefix before the pheno name in the RDS file name', default="/project/ukbblatent/clinicaldata/v1.1.0_seventh_basket/assessmentCentre_82779_MD_13_06_2024_12_18_59.tsv")
    parser.add_argument('--cov_root', action="store", default='/group/glastonbury/GWAS/inputs/covariates/wholeCohort') 
    parser.add_argument('--out_root', action="store", default='/project/ukbblatent/clinicaldata/Atlas/binary_disease_cohorts/WholeCohort_prova') 

    parser.add_argument('--codes', action="store", default='Phecode', help='Comma-seperated list of codes to consider. Options: [Phecode,PhecodeCh,PhecodeLvl1]') 
    parser.add_argument('--modes', action="store", default='near', help='Comma-seperated list of codes to consider. Options: [all,diag,diag10y,diag5y,prog,prog5y,near,near5y,near2y]') 
    parser.add_argument('--cov_suffix', action="store", default='woSmoking', help='Suffix for covariate files') 

    parser.add_argument('--cutoff_date', type=str, help='To filter abnormal dates, a cutoff date must be used. Any entry in the future wrt this date will be removed.', default='2023-10-31')
    
    args, unknown_args = parser.parse_known_args()

    return args, unknown_args

def main():
    args, _ = process_arguments()
    
    # all3Phe = pd.read_csv(args.processed_phecodes, low_memory=False)
    # all3Phe['date'] = pd.to_datetime(all3Phe['date'], format='%Y-%m-%d', errors='coerce')
    # imSubs = get_whole_cohort(cov_root=args.cov_root)

    # print("\nDisease cohorts:")
    # for k,v in imSubs.items():
    #     print(f"\n{k}:")
    #     print(all3Phe[all3Phe.eid.isin(v)][['Phecode', 'PhecodeCh', 'PhecodeLvl1']].nunique())

    # cov_cohorts, dis_cohorts, disnodis_cohorts = create_cohorts(args.cov_root, args.info_assessment_centre_path, imSubs, all3Phe, cutoff_date=args.cutoff_date, cov_suffix=args.cov_suffix)
    
    # os.makedirs(args.out_root, exist_ok=True)

    # with open(f"{args.out_root}/raw_cov_cohorts.pkl", "wb") as f:
    #     pickle.dump(cov_cohorts, f)
    # with open(f"{args.out_root}/raw_dis_cohorts.pkl", "wb") as f:
    #     pickle.dump(dis_cohorts, f)
    # with open(f"{args.out_root}/raw_disnodis_cohorts.pkl", "wb") as f:
    #     pickle.dump(disnodis_cohorts, f)

    #read existing ones
    with open(f"{args.out_root}/raw_disnodis_cohorts.pkl", "rb") as f:
        disnodis_cohorts = pickle.load(f)

    codes = args.codes.split(",")
    modes = args.modes.split(",")
    
    for cohort in ["wholeCohort", "wholeCohort_caucasian"]:
        print(f"\ncohort: {cohort}:-----------------")
        for code_col in codes:
            print(f"\nCode col: {code_col}:-----------------")
            for mode in modes:
                print(f"\nMode: {mode}:-----------------")
                os.makedirs(f"{args.out_root}/{cohort}/{code_col}/{mode}", exist_ok=True)
                disDFs = create_dis_cohorts(disnodis_cohorts[cohort], mode, code_col)
                for c, df in disDFs.items():
                    df.reset_index().to_csv(f"{args.out_root}/{cohort}/{code_col}/{mode}/{c}.csv", index=False)  


if __name__ == "__main__":
    main()