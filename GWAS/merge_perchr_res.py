import argparse
import os
import pandas as pd
from glob import glob
from tqdm import tqdm

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path2gwasout', type=str, default="/group/glastonbury/soumick/Exome/results/F20208v3_DiffAE_select_latents_r80_discov/EWAS/run2_MAC1_GWASCond_gwsig_allSNPsFORallpheno_step1woXom_regenie_step2_results", help='Location where the GWAS/EWAS results are storred in per-chromosome manner. ')
    parser.add_argument('--res_type', type=str, default="ewas", help='Set it to the type of result, gwas or ewas (default) or burden')
    
    parser.add_argument('--path4res', type=str, default="", help='Where chromosome-merged sumstats will be storred [If blank, it will be storred inside the /results/<res_type> inside path2gwasout - to be able to use other scripts without change]')
    parser.add_argument('--prefix2remove', type=str, default="ewas_regenie_", help='Prefix to remove from the input file names for creating output file names')
    parser.add_argument('--tag2add', type=str, default="ewas", help='Tag to add after the name of the actual file, within dots [e.g. .ewas]')

    parser.add_argument("--gwas_sep_space", action=argparse.BooleanOptionalAction, default=True, help="Whether to GWAS used the space as seperator instead of tab (e.g. older regenie). [Output will be saved as tab-seperated file]")
    parser.add_argument("--chr_X", action=argparse.BooleanOptionalAction, default=False, help="Whether to process chromosome X as well. [Default: False]")
    parser.add_argument("--chr_Y", action=argparse.BooleanOptionalAction, default=False, help="Whether to process chromosome X as well. [Default: False]")
    
    parser.add_argument("--save_all_pheno", action=argparse.BooleanOptionalAction, default=True, help="Whether to save sumstats from all pheno in a single file, in addition to the individual ones. [Default: True]")

    return parser

if __name__ == "__main__":
    parser = getARGSParser()
    args, _ = parser.parse_known_args() 

    if args.path4res == "":
        args.path4res = f"{args.path2gwasout}/results/{args.res_type}"
    os.makedirs(args.path4res, exist_ok=True)

    gwas_outs_chr1 = glob(f"{args.path2gwasout}/*_chr1_*.gz") #Reading only chr1 files and then will traverse through all the subsequent chromosomes

    all_pheno = []

    for gwas_out_chr1 in tqdm(gwas_outs_chr1):
        gwas_out_merged = os.path.basename(gwas_out_chr1).replace("_chr1_", "_").replace(args.prefix2remove, "")
        if bool(args.tag2add):
            gwas_out_merged_parts = gwas_out_merged.split(".")
            gwas_out_merged_parts.insert(1, args.tag2add)
            gwas_out_merged = ".".join(gwas_out_merged_parts)

        print(f"Processing {gwas_out_merged}................\n")

        print("Processing chromosome 1..\n")
        df = pd.read_csv(gwas_out_chr1, sep=" " if args.gwas_sep_space else "\t")

        for ch in range(2, 23):
            print(f"Processing chromosome {ch}..\n")
            gwas_out_chr = gwas_out_chr1.replace("_chr1_", f"_chr{ch}_")
            df = pd.concat([df, pd.read_csv(gwas_out_chr, sep=" " if args.gwas_sep_space else "\t")], ignore_index=True)

        if args.chr_X:
            print("Processing chromosome X..\n")
            gwas_out_chr = gwas_out_chr1.replace("_chr1_", "_chrX_")
            df = pd.concat([df, pd.read_csv(gwas_out_chr, sep=" " if args.gwas_sep_space else "\t")], ignore_index=True)

        if args.chr_Y:
            print("Processing chromosome Y..\n")
            gwas_out_chr = gwas_out_chr1.replace("_chr1_", "_chrY_")
            df = pd.concat([df, pd.read_csv(gwas_out_chr, sep=" " if args.gwas_sep_space else "\t")], ignore_index=True)
            
        df.to_csv(f"{args.path4res}/{gwas_out_merged}", sep="\t", index=False, compression="gzip")  

        if args.save_all_pheno:
            df['pheno'] = gwas_out_merged.split(".")[0]
            all_pheno.append(df)

    if args.save_all_pheno:
        all_pheno = pd.concat(all_pheno, ignore_index=True)
        all_pheno.to_pickle(f"{os.path.dirname(args.path4res)}/all_pheno.{args.tag2add}.merged.pkl")
        # all_pheno.to_csv(f"{os.path.dirname(args.path4res)}/all_pheno.{args.tag2add}.merged.gz", sep="\t", index=False, compression="gzip")