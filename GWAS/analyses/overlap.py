import argparse
from glob import glob
import pandas as pd
import os
from tqdm import tqdm
import re
import pickle
import numpy as np
import logging

from utils.toploci_merger import adjust_merged_loci
from utils.summarise import summarise_singlerun, summarise_multirun

def create_logger(name, file):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def clean_sp2(value):
    value = re.sub(r'\(.*?\)', '', value)
    value = value.replace('NONE', '')
    return value

def clean_finalDF(df):
    df.replace('NA', np.nan, inplace=True)
    conditions = [
        (df['runA'].notna() & df['runB'].notna()),  # Priority 1
        (df['runA'].isna() & df['runB'].notna()),   # Priority 2
        (df['runA'].notna() & df['runB'].isna())   # Priority 3
    ]
    choices = [1, 2, 3]
    df['Priority'] = np.select(conditions, choices, default=np.nan)
    df.sort_values(['SNP', 'Priority'], ascending=[True, True], inplace=True)
    df.drop_duplicates(subset='SNP', keep='first', inplace=True)
    df.drop(columns='Priority', inplace=True)
    df.fillna('NA', inplace=True)

    return df

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path2gwasouts', type=str, default="/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3", help='Coma-seperated list of GWAS runs to compare. If only one run is provided, the script will only merge the toploci files and create all_toploci_merged.tsv.')
    parser.add_argument("--processed_multirun_tsvs", type=str, default="", help="Can directly supply processed multirun TSVs and then the path2gwasouts param will be ignored.")
    
    parser.add_argument('--path4res', type=str, default="/group/glastonbury/soumick/GWAS/overlaps", help='Path where the combo results will be saved. [Will be ignored if only one run (path2gwasouts) is provided]')
    parser.add_argument('--save_tag', type=str, default="5Seeds_FVAEvsVAEvsUVAE3Ph2LDo50", help='Tag to identify this comparison [Will be ignored if only one run (path2gwasouts) is provided]')
    
    parser.add_argument('--filter_toploci_mode', type=int, default=0, help="Mode of filtering the GWAS tophits based on the toploci tables. [0: Don't filter, 1: Filter based on the Total number of other SNPs in clump (col: TOTAL), 2: Filter based on the Number of clumped SNPs p < 0.0001 (col: S0001)]")
    parser.add_argument('--min_toploci', type=int, default=1, help="For filter_toploci mode 1 or 2, what should be the minimum value to be considered")
    
    parser.add_argument('--filter_n_min_runs', type=int, default=1, help='An addtional filter, post-processing after obtaining the earlier results. In how many runs a loci must be present to be considered in this filtered one. [Default: 1, i.e. no filtering]')    
    
    return parser

if __name__ == "__main__":
    args = getARGSParser().parse_args()

    if bool(args.processed_multirun_tsvs):
        paths2process = args.processed_multirun_tsvs.split(",")
    else:
        paths2process = args.path2gwasouts.split(",")    

    merged_toploci = []
    for path2process in paths2process:
        runname = path2process.split("/")[-1]
        print(f"Processing {runname}....")
        if bool(args.processed_multirun_tsvs):
            runname = runname.replace("all_toploci_multirun_", "").replace("_merged.tsv", "")
            paths_toplocis = [path2process]
            if len(paths2process) > 1:
                logger = create_logger(f"log_{path2process}", f"{args.path4res}/log_merged_analysis_{args.save_tag}.log")
            else:
                logger = create_logger(f"log_{path2process}", f"{os.path.dirname(path2process)}/all_toploci_merged.log")
        else:
            path2process = f"{path2process.strip()}/results/gwas/toploci"
            paths_toplocis = glob(f"{path2process}/**/*.toploci.tsv", recursive=True)
            logger = create_logger(f"log_{path2process}", f"{path2process}/all_toploci_merged.log")

        toplocis = []
        for t in tqdm(paths_toplocis):
            df = pd.read_table(t)
            if args.filter_toploci_mode == 1:
                df = df[df['TOTAL'] >= args.min_toploci]
            elif args.filter_toploci_mode == 2:
                df = df[df['S0001'] >= args.min_toploci]
            if len(df) > 0:
                if bool(args.processed_multirun_tsvs):
                    df['Run'] = runname
                else:
                    df['Pheno'] = os.path.basename(t).replace(".toploci.tsv", "")
                toplocis.append(df)

        toplocis = pd.concat(toplocis)
        for snp in toplocis['SNP']:
            if toplocis['SP2'].str.contains(snp).any():
                logger.warning(f"SNP_SP2_Warning (Before filtering): SNP {snp} is present in another SP2! Decrease the number of toploci when counting!")
        if not bool(args.processed_multirun_tsvs):
            toplocis.to_csv(f"{path2process}/all_toploci_merged_raw.tsv", sep="\t", index=False)
        
        toplocis = adjust_merged_loci(toplocis, loci_radius=500000)
        if not bool(args.processed_multirun_tsvs):
            toplocis.to_csv(f"{path2process}/all_toploci_merged.tsv", sep="\t", index=False)    
            summarise_singlerun(f"{path2process}/all_toploci_merged.tsv", f"{path2process}/all_toploci_merged_summary.txt")
        for snp in toplocis['SNP']:
            if toplocis['SP2'].str.contains(snp).any():
                logger.warning(f"SNP_SP2_Warning (After filtering): SNP {snp} is present in another SP2! Decrease the number of toploci when counting!")

        toplocis['Run'] = runname
        merged_toploci.append(toplocis)

    if len(paths2process) > 1:
        os.makedirs(f"{args.path4res}/TSVs", exist_ok=True)
        os.makedirs(f"{args.path4res}/logs", exist_ok=True)
        os.makedirs(f"{args.path4res}/summaries", exist_ok=True)        
        
        logger = create_logger(f"log_overlap_{args.save_tag}_{args.path4res}", f"{args.path4res}/logs/overlapfinder_{args.save_tag}.log")

        merged_toploci = pd.concat(merged_toploci)
        
        for snp in merged_toploci['SNP']:
            if merged_toploci['SP2'].str.contains(snp).any():
                logger.warning(f"Multi-run Overlap - SNP_SP2_Warning (Before filtering): SNP {snp} is present in another SP2! Decrease the number of toploci when counting!")
                print(f"Multi-run Overlap - SNP_SP2_Warning (Before filtering): SNP {snp} is present in another SP2! Decrease the number of toploci when counting!")
        merged_toploci.to_csv(f"{args.path4res}/TSVs/all_toploci_multirun_{args.save_tag}_merged_raw.tsv", sep="\t", index=False)

        merged_toploci = adjust_merged_loci(merged_toploci)
        merged_toploci['RunCount'] = merged_toploci['Run'].apply(lambda x: len(x.split(",")))
        
        for snp in merged_toploci['SNP']:
            if merged_toploci['SP2'].str.contains(snp).any():
                logger.warning(f"Multi-run Overlap - SNP_SP2_Warning (After filtering): SNP {snp} is present in another SP2! Decrease the number of toploci when counting!")
                print(f"Multi-run Overlap - SNP_SP2_Warning (After filtering): SNP {snp} is present in another SP2! Decrease the number of toploci when counting!")
        merged_toploci.to_csv(f"{args.path4res}/TSVs/all_toploci_multirun_{args.save_tag}_merged.tsv", sep="\t", index=False)    
        
        summarise_multirun(f"{args.path4res}/TSVs/all_toploci_multirun_{args.save_tag}_merged.tsv", f"{args.path4res}/summaries/all_toploci_multirun_{args.save_tag}_merged.txt")
        
        if args.filter_n_min_runs > 1:
            filtered_toploci = merged_toploci[merged_toploci['RunCount'] >= args.filter_n_min_runs]
            filtered_toploci.to_csv(f"{args.path4res}/TSVs/all_toploci_multirun_{args.save_tag}_merged_min{args.filter_n_min_runs}runs.tsv", sep="\t", index=False)
            summarise_multirun(f"{args.path4res}/TSVs/all_toploci_multirun_{args.save_tag}_merged_min{args.filter_n_min_runs}runs.tsv", f"{args.path4res}/summaries/all_toploci_multirun_{args.save_tag}_merged_min{args.filter_n_min_runs}runs.txt")

    print("Done!")
