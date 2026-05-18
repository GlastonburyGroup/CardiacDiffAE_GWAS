#imports
import argparse
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from dateutil.parser import parse

sys.path.insert(0, os.getcwd()) #to handle the sub-foldered structure of the tricoder

from analyses.latents.utils import get_attributes_ukbbpuller, remove_outliers, compute_age


def process_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pth_prefix', action="store", default="/scratch/glastonbury/datasets/ukbbH5s/F20208_Long_axis_heart_images_DICOM_H5v3/meta") #using H5tools/traverse_DSH5.py (--mode 0)
    parser.add_argument('--pth_ids', action="store", default="subIDs.json") #using H5tools/traverse_DSH5.py (--mode 0)
    parser.add_argument('--pth_MRIdates', action="store", default="subIDs_MRIdates.csv") #using H5tools/traverse_DSH5.py (--mode 5)
    parser.add_argument('--pth_MRIcentre', action="store", default="subIDs_Acqs_MRICentre.json") #using H5tools/traverse_DSH5.py (--mode 6)

    parser.add_argument('--whole_cohort', action=argparse.BooleanOptionalAction, default=False, help="Whether to use the whole cohort (not considering MRI subjects only) or not")
    parser.add_argument('--filename', action="store", default="wholeCohort", help="Name of the file to save the output, only if whole_cohort is True")
    
    parser.add_argument('--outpth', action="store", default="/group/glastonbury/GWAS/inputs/covariates")
    
    parser.add_argument('--data_key_startswith', action="store", default="primary")
    parser.add_argument('--data_keys', action="store", default="", help="Comma-separated list of the data keys to consider (will override the data_key_startswith)")

    parser.add_argument('--drop_MRICentre', action=argparse.BooleanOptionalAction, default=False, help="Whether to drop MRI_Centre (required for T2 FLAIR Brain) [Won't be considered for the whole_cohort]")
    parser.add_argument('--drop_duplicates', action=argparse.BooleanOptionalAction, default=True, help="When a subject has multiple visits, keep only the first one [Won't be considered for the whole_cohort]")
    parser.add_argument('--drop_smoking', action=argparse.BooleanOptionalAction, default=True, help="Whether to drop the smoking column (might gain some subjects)")
    
    parser.add_argument('--compute_BSA', action=argparse.BooleanOptionalAction, default=False, help="Whether to compute the Body Surface Area (might loose some subjects)")
    
    parser.add_argument('--pth_root_clinicaldata', action="store", default="/project/ukbblatent/clinicaldata")
    parser.add_argument('--pth_bodymeas', action="store", default="/project/ukbblatent/clinicaldata/v1.1.0_seventh_basket/phisicalinfo_MD_27_10_2023_22_16_22.tsv", help="path to the body measurements file (only needed if compute_BSA is True)")
    parser.add_argument('--pth_ethnicity', action="store", default="/project/ukbblatent/clinicaldata/v1.1.0_seventh_basket/ethnic_MD_14_11_2023_10_34_13.tsv")
    parser.add_argument('--pth_sexchp', action="store", default="/project/ukbblatent/clinicaldata/v1.1.0_seventh_basket/gensexchip_MD_10_11_2023_15_45_28.tsv")
    parser.add_argument('--pth_exome_tranche', action="store", default="/group/glastonbury/soumick/Exome/ukbb_WES_tranche_release_cov.tsv")
    
    parser.add_argument('--pth_outliers', action="store", default="", help="path to the csv file containing the outlier subIDs - to create NOnoise version of the covariates")

    args, unknown_args = parser.parse_known_args()

    return args, unknown_args

def main():
    args, _ = process_arguments()

    if bool(args.pth_prefix):
        args.pth_ids = f"{args.pth_prefix}/{args.pth_ids}"
        args.pth_MRIdates = f"{args.pth_prefix}/{args.pth_MRIdates}"
        args.pth_MRIcentre = f"{args.pth_prefix}/{args.pth_MRIcentre}"
    
    pth_baseline = f"{args.pth_root_clinicaldata}/baseline.tsv"
    pth_dob = f"{args.pth_root_clinicaldata}/dateofbirth.tsv"

    baseline = pd.read_table(pth_baseline).set_index('f.eid')
    dob = pd.read_table(pth_dob).set_index('f.eid')
    baseline = dob.join(baseline)

    baseline_cols = {'f.34.':   'Birth_Year',
                    'f.52.':    'Birth_Month',
                    'f.31.':    'Gender',
                    'f.20116.': 'CAT_Smoking',
                    'f.21000.': 'CAT_Ethnicity', 
                    'f.21001.': 'BMI', 
                    'f.21022.': 'Age_Recruitment',
                    'f.40000.': 'Death_Date',
                    'f.40001.': 'CAT_Death_Cause',
                    }
                    
    cols_nodrop = ['f.40000.', 'f.40001.']

    if not args.whole_cohort:
        ids = list(np.array(json.load(open(args.pth_ids, 'r')), dtype=int))
    else:
        ids = []

    baseline = get_attributes_ukbbpuller(file_path="", df=baseline, indices=ids, thresh_notna=0.3, cols_nodrop=cols_nodrop, col_names=baseline_cols, custom_dict={}, default_order=[2,3,1,0,4,5,6])
    baseline = baseline.dropna(subset=["Birth_Year", "Birth_Month"])

    sexchp = pd.read_table(args.pth_sexchp).set_index('f.eid')
    sexchp = sexchp.rename(columns={'f.22000.0.0': 'Genotype_Batch', 'f.22001.0.0': 'Sex'}).dropna()
    sexchp['Genotype_Batch'] = sexchp['Genotype_Batch'].apply(lambda x: 1 if x > 0 else 0)
    baseline = baseline.join(sexchp.astype(np.int32), how="inner")

    ethnicity = pd.read_table(args.pth_ethnicity).set_index('f.eid')
    ethnicity = ethnicity[['f.22006.0.0']].rename(columns={'f.22006.0.0': 'isCaucasian'}).fillna(0.0)
    baseline = baseline.join(ethnicity.astype(np.int32), how="inner")

    cols2drop = ['Gender', "Death_Date", "CAT_Death_Cause", "Age_Recruitment",  "Birth_Year", "Birth_Month", "CAT_Ethnicity"]

    if not args.whole_cohort:
        dates = pd.read_csv(args.pth_MRIdates).set_index('eid')
        cov = baseline.join(dates, how="inner")
        cov['Age_MRI'] = cov.apply(compute_age, axis=1, col_year_of_birth="Birth_Year", col_month_of_birth="Birth_Month").values
        cov['instance'] = cov['instance'].apply(lambda x: x[:1]).astype(int)

        cov['Date_MRIvisit'] = cov['date'].apply(lambda x: parse(x).strftime('%Y'))
        cols2drop.append("date")        

        cov_cols = {
            "BMI" : "BMI",
            "Age_MRI": "Age",
            "Date_MRIvisit" : "MRI_Date",
            "instance": "MRI_Visit"
        }
        cov = cov.rename(columns=cov_cols)
    else:
        cov = baseline
    
    if args.drop_smoking:
        cols2drop.append("CAT_Smoking")
    else:
        cov.CAT_Smoking = cov.CAT_Smoking.replace(-3.0, np.nan) #-3 = prefer not to answer

    cov = cov.drop(columns=cols2drop)
    cov.BMI = cov.BMI.astype(float)    
    
    cov.insert(0, 'FID', cov.index)
    cov.insert(1, 'IID', cov.index)
    
    if (not args.whole_cohort) and (not args.drop_MRICentre):
        if not bool(args.pth_MRIcentre) or not os.path.exists(args.pth_MRIcentre):
            print("MRI Centre file (--pth_MRIcentre) not found, even though drop_MRICentre is not set to True! Ignorning MRI Centre and cotinuing...")
            args.drop_MRICentre = True
        else:
            centres = json.load(open(args.pth_MRIcentre, 'r'))
            if bool(args.data_keys):
                keys = args.data_keys.split(',') if ',' in args.data_keys else args.data_keys.split('OR')
            else:
                keys = [key for key in centres.keys() if key.startswith(args.data_key_startswith)]
            print(f"Considering the following keys: {keys}...")

            centres_num = {}
            for view in keys:
                inner_dict = centres[view]
                for address in inner_dict:
                    subjects = inner_dict[address]
                    if address in centres_num:
                        centres_num[address].extend(subjects)
                    else:
                        centres_num[address] = subjects

            centres_dict = {int(subid): i for i, (address, lst) in enumerate(centres_num.items()) for subid in lst}
            cov['MRI_Centre'] = cov.index.map(lambda x: centres_dict.get(x))

    # remove the subjects with duplicates (so both visits) from the subset of IDs with visit = 3 
    if (not args.whole_cohort) and args.drop_duplicates:
        only3 = set(cov[cov.MRI_Visit == 3].index) - set(cov[cov.duplicated(subset=['FID'])].index)
        df2 = cov[cov.MRI_Visit == 2]
        df3 = cov.loc[list(only3)]
        cov = pd.concat([df2,df3])

    cov = cov.dropna()
    cov['BMI'] = round(cov['BMI'],3)

    if args.compute_BSA:
        body_measure_cols ={
            'f.21002.' :'Weight',
            'f.49.'    :'Hip_circumference',
            'f.50.'    :'standing_height',  
            'f.23127.' :'trunk fat %',
            'f.23099.' :'body fat %'
        }
        body_measure_cols2drop = ['f.23099.', 'f.23127.']
        body = get_attributes_ukbbpuller(file_path=args.pth_bodymeas, indices=ids, thresh_notna=0.3, cols2drop=body_measure_cols2drop, col_names=body_measure_cols, custom_dict={}, default_order=[2,3,1,0,4,5,6])
        body = remove_outliers(body, contamination=0.01, support_fraction=0.7)

        body['BSA'] = 0.20247*(body['Weight']**0.425)*((body['standing_height']/100)**0.725)
        cov['BSA'] = cov.index.map(lambda x: body['BSA'].get(x))
        cov['BSA'] = round(cov['BSA'], 3)
        cov = cov.dropna()    
        cov['MRI_Centre'] = cov['MRI_Centre'].astype(int)    

    filename = args.filename if args.whole_cohort else args.pth_ids.split("/meta")[0].split("/")[-1] 
    args.outpth = f"{args.outpth}/{filename}"
    os.makedirs(args.outpth, exist_ok=True)

    filename = filename if not args.compute_BSA else f"{filename}_wBSA"
    filename = filename if not args.drop_smoking else f"{filename}_woSmoking"
    filename = filename if args.whole_cohort or (not args.drop_MRICentre) else f"{filename}_woMRICentre"
    filename = filename if args.whole_cohort or (args.drop_duplicates) else f"{filename}_MultiVisits"
    cov.to_csv(f"{args.outpth}/cov_{filename}.tsv", sep='\t', index=False)

    cov_caucasian = cov[cov['isCaucasian'] == 1]
    cov_caucasian.to_csv(f"{args.outpth}/cov_{filename}_caucasian.tsv", sep='\t', index=False)
    
    if bool(args.pth_outliers):
        outliers = pd.read_csv(args.pth_outliers)
        newcov = cov[~cov['FID'].isin(outliers['subID'])]
        newcov.to_csv(f"{args.outpth}/cov_{filename}_NOnoise.tsv", sep='\t', index=False)

        newcov_caucasian = cov_caucasian[~cov_caucasian['FID'].isin(outliers['subID'])]
        newcov_caucasian.to_csv(f"{args.outpth}/cov_{filename}_caucasian_NOnoise.tsv", sep='\t', index=False)

    exome_tranche = pd.read_table(args.pth_exome_tranche).set_index("FID").dropna()
    exome_tranche['exome_tranche'] = exome_tranche['exome_tranche'].apply(lambda x: 1 if x == "later_releases" else 0)
    cov = cov.join(exome_tranche['exome_tranche'], how="inner")
    cov.to_csv(f"{args.outpth}/cov_{filename}_wXomeTran.tsv", sep='\t', index=False)

    cov_caucasian = cov_caucasian.join(exome_tranche['exome_tranche'], how="inner")
    cov_caucasian.to_csv(f"{args.outpth}/cov_{filename}_caucasian_wXomeTran.tsv", sep='\t', index=False)

    if bool(args.pth_outliers):
        newcov = cov[~cov['FID'].isin(outliers['subID'])]
        newcov.to_csv(f"{args.outpth}/cov_{filename}_wXomeTran_NOnoise.tsv", sep='\t', index=False)

        newcov_caucasian = cov_caucasian[~cov_caucasian['FID'].isin(outliers['subID'])]
        newcov_caucasian.to_csv(f"{args.outpth}/cov_{filename}_caucasian_wXomeTran_NOnoise.tsv", sep='\t', index=False)

    print("Done!")
    
if __name__ == "__main__":
    args, _ = process_arguments()
    main()