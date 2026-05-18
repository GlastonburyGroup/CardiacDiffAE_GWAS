import os
import pandas as pd
import numpy as np
import argparse
import pickle

from tqdm import tqdm

from icdmappings import Mapper

mapper = Mapper()

cov_tag_dit  = {
    "Heart": {
        "20208": "F20208_Long_axis_heart_images_DICOM_H5v3",
        "20214": "F20214_Experimental_shMOLLI_sequence_heart_images_DICOM_H5v3",
    },
    "Pancreas": {
        "20259": "F20259_Pancreas_Images_ShMoLLI_DICOM_H5v3",
        "20260": "F20260_Pancreas_Images_gradient_echo_DICOM_H5v3",},
    "Liver": {
        "20204": "F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3",
        "20254": "F20254_Liver_imaging_IDEAL_protocol_DICOM_H5v3",},
    "Brain": {
        "20250": "F20250_Multiband_diffusion_brain_images_NIFTI_H5v3",
        "20251": "F20251_Susceptibility_weighted_brain_images_NIFTI_H5v3",
        "20253": "F20253_T2_FLAIR_structural_brain_images_NIFTI_H5v3",},
    "DXA": {
        "20158": "F20158_DXA_images_H5v3",},
    # "Eye": {
    #     "21015n6": "",
    #     "21017n8": "",},
}

def get_imaging_cohorts(ds_root="/scratch/glastonbury/datasets/ukbbH5s"):

    imSubs  = {}

    #Whole body-------------------
    #20158
    dxa = set(pd.read_json(f"{ds_root}/{cov_tag_dit['DXA']['20158']}/meta/subIDs_primary_APSpine_0ORprimary_LeftFemur_0ORprimary_LeftOrthoKnee_0ORprimary_LVA_0ORprimary_RightFemur_0ORprimary_RightOrthoKnee_0ORprimary_TotalBodySoftTissue_0ORprimary_TotalBodySkeleton_0.json")[0].to_list())
    print(f"DXA: # from meta: {len(dxa)}")
    imSubs["DXA"] = dxa

    #20201  *********************************** (not working - courrupted?)

    #Liver-------------------
    #20204
    liver_204 = set(pd.read_json(f"{ds_root}/{cov_tag_dit['Liver']['20204']}/meta/subIDs_primary_0.json")[0].to_list())
    print(f"Liver 20204: # from meta: {len(liver_204)}")

    #20254
    liver_254 = set(pd.read_json(f"{ds_root}/{cov_tag_dit['Liver']['20254']}/meta/subIDs_primary_0.json")[0].to_list())
    print(f"Liver 20254: # from meta: {len(liver_254)}")

    liver = liver_204.intersection(liver_254)
    print(f"Liver: {len(liver)}")
    imSubs["Liver"] = liver

    #Pancreas-------------------
    #20259
    pancreas_259 = set(pd.read_json(f"{ds_root}/{cov_tag_dit['Pancreas']['20259']}/meta/subIDs_primary_0.json")[0].to_list())
    print(f"Pancreas 20259: # from meta: {len(pancreas_259)}")

    #20260
    pancreas_260 = set(pd.read_json(f"{ds_root}/{cov_tag_dit['Pancreas']['20260']}/meta/subIDs_primary_0.json")[0].to_list())
    print(f"Pancreas 20260: # from meta: {len(pancreas_260)}")

    pancreas = pancreas_259.intersection(pancreas_260)
    print(f"Pancreas: {len(pancreas)}")
    imSubs["Pancreas"] = pancreas

    #Heart-------------------
    #20208
    heart_208 = set(pd.read_json(f"{ds_root}/{cov_tag_dit['Heart']['20208']}/meta/subIDs_primary_LAX_4Ch_transverse_0.json")[0].to_list())
    print(f"Heart 20208: # from meta: {len(heart_208)}")

    #20214
    heart_214 = set(pd.read_json(f"{ds_root}/{cov_tag_dit['Heart']['20214']}/meta/subIDs.json")[0].to_list())
    print(f"Heart 20214: # from meta: {len(heart_214)}")

    heart = heart_208.intersection(heart_214)
    print(f"Heart: {len(heart)}")
    imSubs["Heart"] = heart

    #Brain-------------------
    #20250
    brain_250 = set(pd.read_json(f"{ds_root}/{cov_tag_dit['Brain']['20250']}/meta/subIDs_primary_FAORprimary_MD.json")[0].to_list())
    print(f"Brain 20250: # from meta: {len(brain_250)}")

    #20251
    brain_251 = set(pd.read_json(f"{ds_root}/{cov_tag_dit['Brain']['20251']}/meta/data_SWI_subIDs_primary_SWI.json")[0].to_list())
    print(f"Brain 20251: # from meta: {len(brain_251)}")

    #20253 
    brain_253 = set(pd.read_json(f"{ds_root}/{cov_tag_dit['Brain']['20253']}/meta/data_T2_FLAIR_brain_subIDs_primary.json")[0].to_list())
    print(f"Brain 20253: # from meta: {len(brain_253)}")

    brain = brain_250.intersection(brain_251).intersection(brain_253)
    print(f"Brain: {len(brain)}")
    imSubs["Brain"] = brain

    #Eye-------------------
    #21015 & 21016 
    eye_fundus = set(pd.read_json(f"{ds_root}/F21015n6_Fundus_retinal_eye_image_leftNright_H5v3/meta/subIDs_primary_leftORprimary_right.json")[0].to_list())
    print(f"Eye 21015: # from meta: {len(eye_fundus)}")

    #21017 & 21018 
    eye_oct = set(pd.read_json(f"{ds_root}/F21017n8_OCT_image_slices_leftNright_H5v3/meta/subIDs_primary_leftORprimary_right.json")[0].to_list())
    print(f"Eye 21018: # from meta: {len(eye_oct)}")

    eye = eye_fundus.intersection(eye_oct)
    print(f"Eye: {len(eye)}")
    imSubs["Eye"] = eye

    all_noeye = heart.intersection(pancreas).intersection(liver).intersection(brain).intersection(dxa)
    print(f"All (wo eye): {len(all_noeye)}")
    imSubs["All_noeye"] = all_noeye

    all = heart.intersection(pancreas).intersection(liver).intersection(brain).intersection(dxa).intersection(eye)
    print(f"All: {len(all)}")
    imSubs["All"] = all

    return imSubs

def remove_subs_mismatchinst(dfs):
    if not dfs:
        return pd.DataFrame(), pd.DataFrame()
    
    # Dynamically determine columns to compare (exclude FID and IID)
    cols_to_compare = dfs[0].columns.drop(['FID', 'IID']).tolist()
    num_dfs = len(dfs)
    
    # Precompute hash for each dataframe's relevant columns
    hashed_dfs = []
    for idx, df in enumerate(dfs):
        # Subset to necessary columns to reduce memory
        df_subset = df[['FID', 'IID'] + cols_to_compare].copy()
        # Vectorised hash computation
        df_subset['_hash'] = pd.util.hash_pandas_object(df_subset[cols_to_compare], index=False)
        df_subset['source'] = idx
        hashed_dfs.append(df_subset)
    
    # Concatenate all dataframes
    full_df = pd.concat(hashed_dfs, ignore_index=True)
    
    # Group by FID and IID to check for presence and consistency
    grouped = full_df.groupby(['FID', 'IID']).agg(
        all_sources=('source', lambda x: x.nunique() == num_dfs),
        same_hash=('_hash', lambda x: x.nunique() == 1)
    ).reset_index()
    
    # Extract keys for identical records
    identical_keys = grouped[(grouped['all_sources']) & (grouped['same_hash'])][['FID', 'IID']]
    
    # Build identical_df using the first occurrence from original data
    identical_df = dfs[0].merge(identical_keys, on=['FID', 'IID'])
    
    # Build discrepancies_df by excluding identical keys from all dataframes
    discrepancies_dfs = []
    for df in dfs:
        # Anti-join to filter out identical records
        discrep = df.merge(identical_keys, on=['FID', 'IID'], how='left', indicator=True)
        discrep = discrep[discrep['_merge'] == 'left_only'].drop(columns='_merge')
        discrepancies_dfs.append(discrep)
    
    discrepancies_df = pd.concat(discrepancies_dfs, ignore_index=True)
    
    return identical_df, discrepancies_df

def create_imaging_cohorts(cov_root, imSubs, all3Phe, remove_mismatch_inst=True, cov_suffix="woSmoking_woMRICentre"):
    #considering only the subjects when we have the same instance for all the modalities

    cov_organs = {}
    dis_organs = {}
    disnodis_organs = {}
    for organ in cov_tag_dit.keys():
        print(organ)
        covs = []
        for field, tag in cov_tag_dit[organ].items():
            cov = pd.read_csv(f"{cov_root}/{tag}/cov_{tag}_{cov_suffix}.tsv", sep="\t")
            covs.append(cov)
        if remove_mismatch_inst:
            cov, _ = remove_subs_mismatchinst(covs)
        else:
            cov = pd.concat(covs)
        cov_organs[organ] = cov[cov.FID.isin(imSubs[organ])].copy()
        dis_organs[organ] = cov_organs[organ].merge(all3Phe[all3Phe.eid.isin(cov_organs[organ].FID)], how="left", left_on="FID", right_on="eid").dropna(subset=["Phecode"])
        dis_organs[organ]['Diag2MRI_Year_Diff'] = (dis_organs[organ]['MRI_Date'] - dis_organs[organ]['year'])
        del dis_organs[organ]["eid"]
        healthy = cov_organs[organ][~cov_organs[organ].FID.isin(dis_organs[organ].FID)].copy()
        healthy["Phecode"] = "00_000.00"
        healthy["PhecodeLvl1"] = "00_000.0"
        healthy["PhecodeCh"] = "00_000"
        healthy["PhecodeString"] = "NoDisease"
        healthy["PhecodeCategory"] = "PseudoHealthy"
        healthy["Diag2MRI_Year_Diff"] = 0
        disnodis_organs[organ] = pd.concat([dis_organs[organ], healthy])    

    print("All_noeye")
    covs = []
    for k in cov_organs.keys():
        covs.append(cov_organs[k])
    if remove_mismatch_inst:
        combo_cov, _ = remove_subs_mismatchinst(covs)
    else:
        combo_cov = pd.concat(covs)
    cov_organs["All_noeye"] = combo_cov[combo_cov.FID.isin(imSubs["All_noeye"])].copy()
    dis_organs["All_noeye"] = cov_organs["All_noeye"].merge(all3Phe[all3Phe.eid.isin(cov_organs["All_noeye"].FID)], how="left", left_on="FID", right_on="eid").dropna(subset=["Phecode"])
    dis_organs["All_noeye"]['Diag2MRI_Year_Diff'] = (dis_organs["All_noeye"]['MRI_Date'] - dis_organs["All_noeye"]['year'])
    del dis_organs["All_noeye"]["eid"]
    healthy = cov_organs["All_noeye"][~cov_organs["All_noeye"].FID.isin(dis_organs["All_noeye"].FID)].copy()
    healthy["Phecode"] = "00_000.00"
    healthy["PhecodeLvl1"] = "00_000.0"
    healthy["PhecodeCh"] = "00_000"
    healthy["PhecodeString"] = "NoDisease"
    healthy["PhecodeCategory"] = "PseudoHealthy"
    healthy["Diag2MRI_Year_Diff"] = 0
    disnodis_organs["All_noeye"] = pd.concat([dis_organs["All_noeye"], healthy])   

    return cov_organs, dis_organs, disnodis_organs

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

def create_dis_cohorts(organDF, mode, code_col, min_subs=100):
    #mode: "all", "diag", "diag10y", "diag5y" "prog", "prog5y" "near", "near5y", "near2y"
    #code_col: "Phecode", "PhecodeCh", "PhecodeLvl1"
    organDF = organDF.rename(columns={"FID": "f.eid"}).drop(columns="IID")
    organDF = organDF.set_index(["f.eid"])
    unique_codes = organDF[code_col].unique()

    disDFs = {}
    for c in tqdm(unique_codes):
        if c.startswith("00_000"):
            continue

        disDF = organDF[organDF[code_col] == c].copy()
        if mode == "diag":
            disDF = disDF[disDF.Diag2MRI_Year_Diff >= 0]
        elif mode == "diag10y":
            disDF = disDF[(disDF.Diag2MRI_Year_Diff >= 0) & (disDF.Diag2MRI_Year_Diff <= 10)]
        elif mode == "diag5y":
            disDF = disDF[(disDF.Diag2MRI_Year_Diff >= 0) & (disDF.Diag2MRI_Year_Diff <= 5)]
        elif mode == "prog":
            disDF = disDF[disDF.Diag2MRI_Year_Diff < 0]
        elif mode == "prog5y":
            disDF = disDF[(disDF.Diag2MRI_Year_Diff < 0) & (disDF.Diag2MRI_Year_Diff >= -5)]
        elif mode == "near":
            disDF = disDF[(disDF.Diag2MRI_Year_Diff <= 5) & (disDF.Diag2MRI_Year_Diff >= -2)]
        elif mode == "near5y":
            disDF = disDF[(disDF.Diag2MRI_Year_Diff <= 5) & (disDF.Diag2MRI_Year_Diff >= -5)]
        elif mode == "near2y":
            disDF = disDF[(disDF.Diag2MRI_Year_Diff <= 2) & (disDF.Diag2MRI_Year_Diff >= -2)]
        disDF = disDF.iloc[
                    disDF.reset_index()  
                    .assign(abs_diff=lambda x: x['Diag2MRI_Year_Diff'].abs())  
                    .groupby(['f.eid'])['abs_diff']  
                    .idxmin()   #we only need the occurence closest to the MRI
             cov_tag_dit   ]
        subIDs = disDF.index.unique()
        if len(subIDs) < min_subs:
            print(f"{c}: Not enough subjects ({len(subIDs)})")
            continue
        disDF["BinCAT_Disease"] = 1

        nondisDF = organDF[organDF.PhecodeCh != disDF.PhecodeCh.iloc[0]].copy()
        nondisDF = nondisDF[~nondisDF.index.isin(subIDs)]
        nondisDF = nondisDF.iloc[
                    nondisDF.reset_index()  
                    .assign(abs_diff=lambda x: x['Diag2MRI_Year_Diff'].abs())  
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
        nondisDF["Diag2MRI_Year_Diff"] = 0
        nondisDF[["date", "diag_type", "ICD", "ICDch", "Flag", "ICDString", "meaning", "code", "code_format", "year"]] = np.nan
        
        disDFs[c] = pd.concat([disDF, nondisDF])

    return disDFs
    
def process_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--processed_phecodes', action="store", default='/project/ukbblatent/clinicaldata/Atlas/mappedPheX/processed_GP_HI_SR_Phecode_ALL.csv') 
    parser.add_argument('--ds_root', action="store", default='/scratch/glastonbury/datasets/ukbbH5s') 
    parser.add_argument('--cov_root', action="store", default='/group/glastonbury/GWAS/inputs/covariates') 
    parser.add_argument('--out_root', action="store", default='/project/ukbblatent/clinicaldata/Atlas/binary_disease_cohorts/MultiOrganV3_prova') 

    parser.add_argument('--organs', action="store", default='Pancreas', help='Comma-seperated list of organs to consider') 
    parser.add_argument('--codes', action="store", default='Phecode', help='Comma-seperated list of codes to consider. Options: [Phecode,PhecodeCh,PhecodeLvl1]') 
    parser.add_argument('--modes', action="store", default='near', help='Comma-seperated list of codes to consider. Options: [all,diag,diag10y,diag5y,prog,prog5y,near,near5y,near2y]') 
    parser.add_argument('--cov_suffix', action="store", default='woSmoking_woMRICentre', help='Suffix for covariate files') 

    parser.add_argument('--remove_mismatch_inst', action=argparse.BooleanOptionalAction, default=True, help="Remove subjects with mismatched instances across modalities")

    args, unknown_args = parser.parse_known_args()

    return args, unknown_args

def main():
    args, _ = process_arguments()
    
    all3Phe = pd.read_csv(args.processed_phecodes, low_memory=False)
    imSubs = get_imaging_cohorts(ds_root=args.ds_root)

    print("\nDisease cohorts:")
    for k,v in imSubs.items():
        print(f"\n{k}:")
        print(all3Phe[all3Phe.eid.isin(v)][['Phecode', 'PhecodeCh', 'PhecodeLvl1']].nunique())

    cov_organs, dis_organs, disnodis_organs = create_imaging_cohorts(args.cov_root, imSubs, all3Phe, remove_mismatch_inst=args.remove_mismatch_inst, cov_suffix=args.cov_suffix)

    # if args.remove_mismatch_inst:
    #     args.out_root = f"{args.out_root}/ignore_mismatch_inst"
    # os.makedirs(args.out_root, exist_ok=True)

    # with open(f"{args.out_root}/raw_cov_organs.pkl", "wb") as f:
    #     pickle.dump(cov_organs, f)
    # with open(f"{args.out_root}/raw_dis_organs.pkl", "wb") as f:
    #     pickle.dump(dis_organs, f)
    # with open(f"{args.out_root}/raw_disnodis_organs.pkl", "wb") as f:
    #     pickle.dump(disnodis_organs, f)

    organs = args.organs.split(",")
    codes = args.codes.split(",")
    modes = args.modes.split(",")
    
    for organ in organs:
        print(f"\nOrgan: {organ}:-----------------")
        for code_col in codes:
            print(f"\nCode col: {code_col}:-----------------")
            for mode in modes:
                print(f"\nMode: {mode}:-----------------")
                os.makedirs(f"{args.out_root}/{organ}/{code_col}/{mode}", exist_ok=True)
                disDFs = create_dis_cohorts(disnodis_organs[organ], mode, code_col)
                for c, df in disDFs.items():
                    df.reset_index().to_csv(f"{args.out_root}/{organ}/{code_col}/{mode}/{c}.csv", index=False)  


if __name__ == "__main__":
    main()