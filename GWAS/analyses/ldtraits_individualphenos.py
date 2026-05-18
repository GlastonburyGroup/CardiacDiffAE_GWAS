# Script to fetch and LD traits from LDLink

import argparse
from glob import glob
import pandas as pd
import os
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import logging

from utils.ldlink_ldtrait import get_ldtrait
from utils.nlp import gen_wordcloud
from tag_each_ldtrait import tag_each_ldtrait

def process_pheno(t, gwas_outs, args):
    logging.debug(f"Processing {t}.....")
    df_top_loci = pd.read_csv(t, sep="\t")
    if len(df_top_loci) == 0:
        return       

    gwas_out = [g for g in gwas_outs if g.endswith(f"{os.path.basename(t).split('.')[0]}.gwas.regenie.gz")][0]
    df_gwas = pd.read_csv(gwas_out, sep="\t", compression="gzip")
    df_gwas = df_gwas[df_gwas["LOG10P"] > (-1 * np.log10(args.sig_level))]
    df_gwas = df_gwas[df_gwas['A1FREQ'] > 0.01]

    if len(df_gwas) == 0:
        return
    
    pheno = os.path.basename(t).replace(".toploci.tsv", "")

    if args.filter_toploci_mode:
        unique_SNPs_toploci = set(df_top_loci[df_top_loci[args.col_toploci] >= args.min_toploci]['SNP'])
        df_gwas = df_gwas[df_gwas['ID'].isin(unique_SNPs_toploci)]

    unique_sigSNPs = set(df_gwas['ID'])
    if nonRS_SNPs := [
        s for s in unique_sigSNPs if not s.startswith("rs")
    ]:
        with open(f"{args.path4res}/{os.path.basename(t).split('.')[0]}_nonRS_IDs.txt", "w") as f:
            f.write("\n".join(nonRS_SNPs))

    unique_sigSNPs_bonf = set(df_gwas[df_gwas["LOG10P"] > (-1 * np.log10(args.sig_level / args.n4bonferroni))]['ID'])

    long_strings = ""
    long_strings_bonf = ""
    dfs = []
    dfs_bonf = []
    for snp in unique_sigSNPs:
        sp2 = list(df_top_loci[df_top_loci.SNP == snp].SP2)
        sp2 = [rs.split("(")[0] for rs in sp2[0].split(",") if rs.startswith("rs")] if len(sp2) else []
        
        while not snp.startswith("rs") and len(sp2):
            logging.warning(f"Warning: Replacing {snp} with {sp2[0]}, as the lead variant does not start with rs.")
            snp = sp2.pop(0)

        if snp.startswith("rs"):
            df_ldtrait = get_ldtrait(snp.split("_")[0], sp2=sp2 if args.check_sp2 else None, use_logger=True)
            if df_ldtrait is None:
                continue
            df_ldtrait['Pheno'] = pheno
            df_ldtrait['SNP'] = snp
            dfs.append(df_ldtrait)
            if snp in unique_sigSNPs_bonf:
                dfs_bonf.append(df_ldtrait)
                savedir = f"{args.path4res}/bonf{args.n4bonferroni}"
            else:
                savedir = f"{args.path4res}"
            df_ldtrait.to_csv(f"{savedir}/{os.path.basename(t).split('.')[0]}_{snp}.ldtrait.tsv", sep="\t", index=False)
            if args.wordcloud:
                long_string, _ = gen_wordcloud(df_ldtrait, df_is_indtraits=True, background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', filename=f"{savedir}/wordclouds/{os.path.basename(t).split('.')[0]}_{snp}.ldtrait.png")
                if bool(long_string):
                    long_strings += long_string
                    if snp in unique_sigSNPs_bonf:
                        long_strings_bonf += long_string
    
    if args.wordcloud and len(unique_sigSNPs):
        gen_wordcloud(pd.DataFrame({"GWAS Trait": [long_strings.strip()]}), background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', filename=f"{args.path4res}/wordclouds/combo_{os.path.basename(t).split('.')[0]}.ldtrait.png")
        if len(unique_sigSNPs_bonf):
            gen_wordcloud(pd.DataFrame({"GWAS Trait": [long_strings_bonf.strip()]}), background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', filename=f"{args.path4res}/bonf{args.n4bonferroni}/wordclouds/combo_{os.path.basename(t).split('.')[0]}.ldtrait.png")

    return long_strings, long_strings_bonf, dfs, dfs_bonf

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path2gwasout', type=str, default="/project/ukbblatent/Out/Results/F20208_heart_1Ses_time2slc_MskCrop128_V2_3D100ep_L1_4ChTrans128fold0_prec32_pythaemodel-rhvae/GWAS_fullDS/Qntl_WBRIT_time2slc_Msk_V2_3D100ep_L1_128RHVAE_FP32_fullDS", help='Location where the GWAS outputs are storred [The main folder, containing a subfolder results, containing subfolder gwas]')
    parser.add_argument('--path4res', type=str, default="", help='Path where LDLink_LDtraits folder will be created and the outputs will be storred [If blank, it will be storred inside the analyses folder present in the parent folder of path2gwasout]')
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=False, help="If set to True and there are multiple GWAS outs to plot, the plotting will be parallelised.")
    parser.add_argument("--wordcloud", action=argparse.BooleanOptionalAction, default=True, help="Generate wordcloud?.")
    parser.add_argument("--check_sp2", action=argparse.BooleanOptionalAction, default=True, help="Whether to check the related variants from the toploci's SP2 column.")

    parser.add_argument("--sig_level", type=float, default=5e-8, help="Significance level.")
    parser.add_argument("--n4bonferroni", type=float, default=128, help="N runs, for Bonferroni correction.")
    
    parser.add_argument('--filter_toploci_mode', type=int, default=0, help="Mode of filtering the GWAS tophits based on the toploci tables. [0: Don't filter, 1: Filter based on the Total number of other SNPs in clump (col: TOTAL), 2: Filter based on the Number of clumped SNPs p < 0.0001 (col: S0001)]")
    parser.add_argument('--min_toploci', type=int, default=1, help="For filter_toploci mode 1 or 2, what should be the minimum value to be considered")
    
    return parser

if __name__ == "__main__":
    parser = getARGSParser()
    args, _ = parser.parse_known_args() 

    args.path2gwasout = f"{args.path2gwasout}/results/gwas"
    gwas_outs = glob(f"{args.path2gwasout}/*.gz")
    top_locis = glob(f"{args.path2gwasout}/toploci/**/*.toploci.tsv", recursive=True) # Old runs have gwas/toploci/toploci/Zx.toploci.tsv, new runs have gwas/toploci/Zx.toploci.tsv. This handles both.

    if not bool(args.path4res):
        args.path4res = f"{os.path.dirname(args.path2gwasout)}/analyses"
    args.path4res = f"{args.path4res}/LDLink_LDtraits/support_files"

    if args.filter_toploci_mode:
        if args.filter_toploci_mode == 1:
            args.col_toploci = "TOTAL"
        elif args.filter_toploci_mode == 2:
            args.col_toploci = "S0001"
        args.path4res += f"_filtTopLoci_{args.min_toploci}{args.col_toploci}" 

    os.makedirs(args.path4res, exist_ok=True)
    logging.basicConfig(filename=args.path4res.replace("support_files","ldtraits.log"), filemode='w', format='%(asctime)s - %(message)s', level=logging.DEBUG)
    
    if args.wordcloud:
        os.makedirs(f"{args.path4res}/wordclouds", exist_ok=True)
            
    if args.n4bonferroni > 1:
        os.makedirs(f"{args.path4res}/bonf{args.n4bonferroni}", exist_ok=True)
        if args.wordcloud:
            os.makedirs(f"{args.path4res}/bonf{args.n4bonferroni}/wordclouds", exist_ok=True)

    if args.parallel:
        if 'SLURM_JOB_ID' in os.environ:
            n_proc = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            n_proc = mp.cpu_count()
    else:
        n_proc = 1
    n_proc_ = min(n_proc, len(top_locis))

    long_strings = ""
    long_strings_bonf = ""
    dfs = []
    dfs_bonf = []

    if n_proc_ > 1:            
        logging.debug(f"Using {n_proc_} processes to parallelise the plotting.")
        with mp.Pool(processes=n_proc_) as pool:
            func = partial(process_pheno, gwas_outs=gwas_outs, args=args) 
            results = pool.map(func, top_locis) 
            for r in results:
                if r is None:
                    continue
                long_strings += r[0]
                long_strings_bonf += r[1]   
                dfs.extend(r[2])
                dfs_bonf.extend(r[3])
    else:
        for t in tqdm(top_locis):
            res = process_pheno(t, gwas_outs=gwas_outs, args=args)
            if res is None:
                continue
            long_strings += res[0]
            long_strings_bonf += res[1]
            dfs.extend(res[2])
            dfs_bonf.extend(res[3])

    if args.wordcloud and len(long_strings):
        gen_wordcloud(pd.DataFrame({"GWAS Trait": [long_strings.strip()]}), background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', filename=f"{args.path4res}/wordclouds/combo_all.ldtrait.png")
        if len(long_strings_bonf):
            gen_wordcloud(pd.DataFrame({"GWAS Trait": [long_strings_bonf.strip()]}), background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', filename=f"{args.path4res}/bonf{args.n4bonferroni}/wordclouds/combo_all.ldtrait.png")

    if len(dfs):
        df = pd.concat(dfs)
        df.to_csv(args.path4res.replace("support_files", "all.ldtrait.tsv"), sep="\t", index=False)
        tag_each_ldtrait(args.path4res.replace("support_files", "all.ldtrait.tsv"), args.sig_level)

        if args.wordcloud:
            gen_wordcloud(df, df_is_indtraits=True, background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', width=5000, height=5000, filename=args.path4res.replace("support_files", "all.ldtrait.png"))

            df_sig = df[df["P-value"] < args.sig_level]
            gen_wordcloud(df_sig, df_is_indtraits=True, background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', width=5000, height=5000, filename=args.path4res.replace("support_files", "all_sig.ldtrait.png"))
            
            df_sig_R5 = df_sig[df_sig["R2"] >= 0.5]
            gen_wordcloud(df_sig_R5, df_is_indtraits=True, background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', width=5000, height=5000, filename=args.path4res.replace("support_files", "all_sig_R2gt0p5.ldtrait.png"))

            df_sig_R6 = df_sig[df_sig["R2"] >= 0.6]
            gen_wordcloud(df_sig_R6, df_is_indtraits=True, background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', width=5000, height=5000, filename=args.path4res.replace("support_files", "all_sig_R2gt0p6.ldtrait.png"))

            df_sig_R8 = df_sig[df_sig["R2"] >= 0.8]
            gen_wordcloud(df_sig_R8, df_is_indtraits=True, background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', width=5000, height=5000, filename=args.path4res.replace("support_files", "all_sig_R2gt0p8.ldtrait.png"))

    if len(dfs_bonf):
        df_bonf = pd.concat(dfs_bonf)
        df_bonf.to_csv(args.path4res.replace("support_files", f"all_bonf{args.n4bonferroni}.ldtrait.tsv"), sep="\t", index=False)

        if args.wordcloud:
            df_sig = df_bonf[df_bonf["P-value"] < args.sig_level]
            gen_wordcloud(df_sig, df_is_indtraits=True, background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', width=5000, height=5000, filename=args.path4res.replace("support_files", f"all_bonf{args.n4bonferroni}_sig.ldtrait.png"))

            df_sig_R6 = df_sig[df_sig["R2"] >= 0.6]
            gen_wordcloud(df_sig_R6, df_is_indtraits=True, background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', width=5000, height=5000, filename=args.path4res.replace("support_files", f"all_bonf{args.n4bonferroni}_sig_R2gt0p6.ldtrait.png"))