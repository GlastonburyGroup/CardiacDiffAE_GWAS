# Script to fetch and LD traits from LDLink

import argparse
from glob import glob
import pandas as pd
import os
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import shutil

from utils.ldlink_ldtrait import get_ldtrait_multiSNP, get_ldtrait_singleSNP
from utils.nlp import gen_wordcloud
from tag_each_ldtrait import tag_each_ldtrait

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--toplocifile', type=str, default="", help='If supplied, then instead of looking for a set of results from the GWAS results folder, just this will be used [Default: Blank]')
    parser.add_argument('--path4res', type=str, default="", help='Path where LDLink_LDtraits folder will be created and the outputs will be storred [If blank, it will be storred inside the analyses folder present in the parent folder of path2gwasout]')
    parser.add_argument("--wordcloud", action=argparse.BooleanOptionalAction, default=True, help="Generate wordcloud?.")
    parser.add_argument("--indiwordcloud", action=argparse.BooleanOptionalAction, default=False, help="Generate wordcloud for each phenotype-SNP?.")
    parser.add_argument("--check_sp2", action=argparse.BooleanOptionalAction, default=True, help="Whether to check the related variants from the toploci's SP2 column.")
    parser.add_argument("--only_fetch_lead", action=argparse.BooleanOptionalAction, default=True, help="Whether to only search for the lead variant.")
    parser.add_argument("--additive", action=argparse.BooleanOptionalAction, default=True, help="If true, then it will check and remove the already-fetched loci.")

    parser.add_argument("--sig_level", type=float, default=5e-8, help="Significance level.")
    parser.add_argument("--n4bonferroni", type=float, default=128, help="N runs, for Bonferroni correction.")
    
    parser.add_argument('--filter_toploci_mode', type=int, default=0, help="Mode of filtering the GWAS tophits based on the toploci tables. [0: Don't filter, 1: Filter based on the Total number of other SNPs in clump (col: TOTAL), 2: Filter based on the Number of clumped SNPs p < 0.0001 (col: S0001)]")
    parser.add_argument('--min_toploci', type=int, default=1, help="For filter_toploci mode 1 or 2, what should be the minimum value to be considered")
    
    return parser

if __name__ == "__main__":
    parser = getARGSParser()
    args, _ = parser.parse_known_args() 

    if not args.toplocifile.endswith("tsv"):
        if not args.path4res:
            args.path4res = f"{args.toplocifile}/results/analyses"
        args.toplocifile = glob(f"{args.toplocifile}/results/gwas/toploci/**/all_toploci_merged.tsv", recursive=True)[0]
    elif not args.path4res:
        SystemError("Please provide the path4res argument when a specific toploci file has been supplied (i.e. ending with tsv).")
    df_top_loci = pd.read_csv(args.toplocifile, sep="\t")
    
    args.path4res = f"{args.path4res}/LDLink_LDtraits/support_files"

    if args.filter_toploci_mode:
        if args.filter_toploci_mode == 1:
            args.col_toploci = "TOTAL"
        elif args.filter_toploci_mode == 2:
            args.col_toploci = "S0001"
        args.path4res += f"_filtTopLoci_{args.min_toploci}{args.col_toploci}" 

    if args.additive:
        if os.path.isfile(args.path4res.replace("support_files", "all.ldtrait.tsv")):
            df = pd.read_csv(args.path4res.replace("support_files", "all.ldtrait.tsv"), sep="\t")
        else:
            df = pd.DataFrame()
            args.additive = False
    else:
        df = pd.DataFrame()
    
    if not args.additive:
        if os.path.exists(args.path4res.replace("/support_files", "")): #delete the folder if it exists
            shutil.rmtree(args.path4res.replace("/support_files", ""), ignore_errors=True)
        os.makedirs(args.path4res, exist_ok=True)
    
    if args.wordcloud:
        os.makedirs(f"{args.path4res}/wordclouds", exist_ok=True)

    unique_sigSNPs = set(df_top_loci.SNP)
    if args.additive:
        unique_sigSNPs = unique_sigSNPs.difference(set(df.leadSNP))

    dfs = [df]
    for snp in tqdm(unique_sigSNPs):
        sp2 = list(df_top_loci[df_top_loci.SNP == snp].SP2)
        sp2 = [rs.split("(")[0] for rs in sp2[0].split(",") if rs.startswith("rs")] if len(sp2) else []

        if snp.startswith("rs"):
            snps = [snp] + sp2
        else:
            snps = sp2

        if len(snps):
            if "RunPheno" in df_top_loci.columns:
                pheno = f"ind{df_top_loci[df_top_loci.SNP == snp].index.to_series().values[0]}_" #as RunPheno can be quite long, it sometime gives us "filename too long" error.
            elif "Pheno" in df_top_loci.columns:
                pheno = ",".join(df_top_loci[df_top_loci.SNP == snp].Pheno) + "_"
            else:
                pheno = ""

            snps = [s.split("_")[0] for s in snps]

            if not args.only_fetch_lead:
                df_ldtrait = get_ldtrait_multiSNP(snps=snps)
            else:
                while not snp.startswith("rs") and len(sp2):
                    print(f"Warning: Replacing {snp} with {sp2[0]}, as the lead variant does not start with rs.")
                    snp = sp2.pop(0)
                df_ldtrait = get_ldtrait_singleSNP(snp=snp.split("_")[0], sp2=sp2 if args.check_sp2 else None)
            if df_ldtrait is None:
                continue

            df_ldtrait['leadSNP'] = snp
            dfs.append(df_ldtrait)
            savedir = f"{args.path4res}"
            df_ldtrait.to_csv(f"{savedir}/{pheno}{snp}.ldtrait.tsv", sep="\t", index=False)
            if args.indiwordcloud:
                gen_wordcloud(df_ldtrait, df_is_indtraits=True, background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', filename=f"{savedir}/wordclouds/{pheno}{snp}.ldtrait.png")
    
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