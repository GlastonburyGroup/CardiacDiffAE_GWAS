import argparse
import os
from collections import defaultdict
from tqdm import tqdm

import pandas as pd
from bgen import BgenReader

def getARGSParser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--bgen_path", type=str, default="/scratch/edoardo.giacopuzzi/UKBB/step2_dataset/step2_dataset_autosomes.mac100.bgen", help="Path to the bgen file")
    
    parser.add_argument("--select_SNP_path", type=str, default="/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/gwas/indep_sara/pbonf_maf01/gw_sig_snps_post_cojo.csv", help="Path to the bgen file")
    parser.add_argument("--cols", type=str, default="SNP,EA", help="coma-seperated column names for the SNP ID and the effect allele (default: SNP,refA)")

    parser.add_argument("--out_folder", type=str, default="/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/gwas/indep_sara/pbonf_maf01", help="Folder to store the output file in tsv format (same name as the original file will be used) with the dosages (can be directly used with REGENIE)")
    parser.add_argument("--out_file", type=str, default="dosage_gw_sig_snps_post_cojo.tsv", help="Name of the output file. Leave it blank if the original BGEN file name should be used")
    
    return parser

if __name__ == "__main__":
    parser = getARGSParser()
    args, unknown_args = parser.parse_known_args()

    data_dict = defaultdict(dict)

    if bool(args.select_SNP_path):
        select_SNPs_cols = args.cols.split(",")
        select_SNPs = pd.read_csv(args.select_SNP_path)[select_SNPs_cols].drop_duplicates()

    with BgenReader(args.bgen_path, delay_parsing=True) as bfile:
        sampleIDs = bfile.samples
        
        for var in tqdm(bfile):
            rsid = var.rsid
            if bool(args.select_SNP_path):
                matching_rsID = select_SNPs[select_SNPs_cols[0]].apply(lambda x: rsid.startswith(x))
                matching_allele = select_SNPs[select_SNPs_cols[1]].apply(lambda x: rsid.endswith(x))
                if not (matching_rsID & matching_allele).any():
                    continue
            dosages = var.minor_allele_dosage
            for sampleID, dosage in zip(sampleIDs, dosages):
                FID, IID = sampleID.split("_")
                data_dict[(FID, IID)][rsid] = dosage
                
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['FID', 'IID'] + df.columns[2:].tolist()

    if bool(args.out_file):
        df.to_csv(f"{args.out_folder}/{args.out_file}", sep="\t", index=False)
    else:
        df.to_csv(f"{args.out_folder}/{os.path.basename(args.bgen_path).replace('.bgen', '.tsv')}", sep="\t", index=False)

    print("Live long and prosper!")