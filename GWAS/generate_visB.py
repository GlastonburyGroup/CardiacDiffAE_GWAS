import argparse
from glob import glob
import gzip
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import reduce, partial

import gwaslab as gl #https://cloufield.github.io/gwaslab/

def quick_combo_merge_max(df1, df2):
    merged = pd.merge(df1, df2, on=['ID','CHROM','GENPOS'], how='outer')
    merged['LOG10P'] = merged[['LOG10P_x', 'LOG10P_y']].max(axis=1)
    return merged.drop(['LOG10P_x', 'LOG10P_y'], axis=1)

def quick_combo_merge_sigbased(sig_level, df1, df2):
    temp = pd.concat([df1, df2])    
    condition = temp["LOG10P"] > (-1 * np.log10(sig_level))
    merged_high = temp[condition].groupby(['ID','CHROM','GENPOS'], as_index=False).agg({"LOG10P": 'max'})
    merged_low = temp[~condition].drop_duplicates(subset=['ID','CHROM','GENPOS'])
    return pd.concat([merged_high, merged_low], ignore_index=True)

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path2gwasout', type=str, default="", help='Location where the GWAS outputs are storred [The main folder, containing a subfolder results, containing subfolder gwas]')
    parser.add_argument('--path4res', type=str, default="", help='Path where the genetic correlation outputs will be storred [If blank, it will be storred inside the vis folder present in the parent folder of path2gwasout]')
    parser.add_argument('--process_dir_tophits', action=argparse.BooleanOptionalAction, default=False, help="Whether to process the top hits folder. If not, then the top hits will be calcated on the fly using the GWAS outputs and the sig_level.")
    parser.add_argument('--filter_alfreq', action=argparse.BooleanOptionalAction, default=True, help="Whether to filter the GWAS tophits based on the allele frequency (>0.01). [Only if process_dir_tophits is False]")
    parser.add_argument('--filter_toploci_mode', type=int, default=0, help="Mode of filtering the GWAS tophits based on the toploci tables. [0: Don't filter, 1: Filter based on the Total number of other SNPs in clump (col: TOTAL), 2: Filter based on the Number of clumped SNPs p < 0.0001 (col: S0001)]")
    parser.add_argument('--min_toploci', type=int, default=1, help="For filter_toploci mode 1 or 2, what should be the minimum value to be considered")
    parser.add_argument("--plot_only_hits", action=argparse.BooleanOptionalAction, default=True, help="Whether to visualise only the top hits.")
    parser.add_argument("--sig_level", type=float, default=5e-8, help="Significance level.")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True, help="If set to True and there are multiple GWAS outs to plot, the plotting will be parallelised.")
    
    parser.add_argument("--stratify_qq", action=argparse.BooleanOptionalAction, default=True, help="If set to True, the QQ plots will be stratified following the provided bins.")
    parser.add_argument("--qq_bins", type=str, default="0.01,0.05,0.1,0.25,0.5,1.0", help="Bins for stratifying the QQ plots.")
    parser.add_argument("--colour_qq_bins", type=str, default="#f05f4e,#f0ad4e,#5cb85c,#5bc0de,#000042", help="Bins for stratifying the QQ plots.")

    parser.add_argument("--colours", type=str, default= "#597FBD,#74BAD3", help="Two colours for the manhatten plot.")
    parser.add_argument("--colour_sigline", default="blue", help="Colour of the significance line.")
    parser.add_argument("--colour_bonfline", default="red", help="Colour of the Bonferroni line.")
    parser.add_argument("--marker_size", type=str, default= "5,10", help="Marker size.")

    parser.add_argument("--dpi", type=int, default=300, help="DPI of the plots.")
    parser.add_argument("--ext", default="png", help="Saving extension.")

    parser.add_argument("--indi_plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to plot the individual plots for each latent.")
    parser.add_argument("--combo_plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to plot a combo plot, combining all hits in one.")

    #Only for the combo plots
    parser.add_argument("--limited_col_merge", action=argparse.BooleanOptionalAction, default=True, help="If True, only required columns will be loaded from the gz files (only them will be storred in the merged npy and pickle file).")
    parser.add_argument("--quick_merge_mode", default=3, type=int, help="0: quick merge won't be used, 1: quick-merge will be by taking max P-value, 2: quick merge will be using by keeping the significant ones and max-P value for the insignificant ones, 3: Ignore less-significant ones.")
    parser.add_argument("--ignore_threshold", default=2, type=int, help="If quick merge mode is 3, then varaints with LOG10P less than this value will be ignored.")
    
    parser.add_argument("--gene_plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to create plots with gene names.")
    parser.add_argument("--chpos_plot", action=argparse.BooleanOptionalAction, default=False, help="Whether to create plots with chromosome positions.")
    parser.add_argument("--snp_plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to create plots with the SNP IDs.")
    
    return parser

#For generating the initial hits (if tophits folder is not utilised)
def process_hits(g, args, processed_tophits):
    with gzip.open(g, 'rt') as f:
        df = pd.read_table(f)
    df_filtered = df[df["LOG10P"] > (-1 * np.log10(args.sig_level))]
    if len(df_filtered) > 0 and args.filter_alfreq:
        df_filtered = df_filtered[df_filtered['A1FREQ'] > 0.01]
    if len(df_filtered) > 0 and args.filter_toploci_mode:
        top_loci = pd.read_table(g.replace(os.path.basename(g), f"toploci/toploci/{os.path.basename(g).split('.')[0]}.toploci.tsv"))
        unique_SNPs_toploci = set(top_loci[top_loci[args.col_toploci] >= args.min_toploci]['SNP'])
        top_loci[top_loci[args.col_toploci] >= args.min_toploci]['SP2'].dropna().str.split(',').apply(lambda x: unique_SNPs_toploci.update(i.split('(')[0] for i in x))
        df_filtered = df_filtered[df_filtered['ID'].isin(unique_SNPs_toploci)]
    if len(df_filtered) > 0: #there are hits
        processed_tophits[os.path.basename(g).split(".")[0]] = df_filtered

#For indi plots
def plot_item(g, args, corrected_sig_level, usecols):
    df = pd.read_table(g, usecols=usecols)
    if args.filter_alfreq:
        df = df[df['A1FREQ'] > 0.01]  
    # if args.filter_toploci_mode: #This is incorrect in my opinion, because non-toploci SNPs are also important for the plots
    #     top_loci = pd.read_table(g.replace(os.path.basename(g), f"toploci/toploci/{os.path.basename(g).split('.')[0]}.toploci.tsv"))
    #     unique_SNPs_toploci = set(top_loci[top_loci[args.col_toploci] >= args.min_toploci]['SNP'])
    #     top_loci[top_loci[args.col_toploci] >= args.min_toploci]['SP2'].dropna().str.split(',').apply(lambda x: unique_SNPs_toploci.update(i.split('(')[0] for i in x))
    #     df = df[df['ID'].isin(unique_SNPs_toploci)]
    mysumstats = gl.Sumstats(df, fmt="regenie", verbose=False)

    if args.gene_plot:
        mysumstats.plot_mqq(
            mode="b", colors = args.colours, scaled=True, sig_level=args.sig_level, sig_line_color=args.colour_sigline,
            suggestive_sig_line=True, suggestive_sig_level=corrected_sig_level,
            suggestive_sig_line_color=args.colour_bonfline, anno="GENENAME", build="19", marker_size=args.marker_size,
            stratified=args.stratify_qq, maf_bins=args.qq_bins, maf_bin_colors=args.colour_qq_bins,
            verbose=False, save=f"{args.path4res}/genename/{os.path.basename(g).split('.')[0]}_mqq.{args.ext}", 
            save_args={"dpi":args.dpi, "facecolor":"white"}
        )
        plt.close()
    
    if args.chpos_plot:
        mysumstats.plot_mqq(
            mode="b", colors = args.colours, scaled=True, sig_level=args.sig_level, sig_line_color=args.colour_sigline,
            suggestive_sig_line=True, suggestive_sig_level=corrected_sig_level,
            suggestive_sig_line_color=args.colour_bonfline, anno=True, 
            stratified=args.stratify_qq, maf_bins=args.qq_bins, maf_bin_colors=args.colour_qq_bins, marker_size=args.marker_size,
            verbose=False, save=f"{args.path4res}/chpos/{os.path.basename(g).split('.')[0]}_mqq.{args.ext}", 
            save_args={"dpi":args.dpi, "facecolor":"white"}
        )
        plt.close()

    if args.snp_plot:
        mysumstats.plot_mqq(
            mode="b", colors = args.colours, scaled=True, sig_level=args.sig_level, sig_line_color=args.colour_sigline,
            suggestive_sig_line=True, suggestive_sig_level=corrected_sig_level,
            suggestive_sig_line_color=args.colour_bonfline, anno="SNPID", marker_size=args.marker_size, 
            stratified=args.stratify_qq, maf_bins=args.qq_bins, maf_bin_colors=args.colour_qq_bins,
            verbose=False, save=f"{args.path4res}/snpID/{os.path.basename(g).split('.')[0]}_mqq.{args.ext}", 
            save_args={"dpi":args.dpi, "facecolor":"white"}
        )
        plt.close()

#For combo plots
def fetch_DF_combomerge(g, args, usecols=None):
    with gzip.open(g, 'rt') as f:
        if args.limited_col_merge and usecols:
            df = pd.read_table(f, usecols=usecols)
        else:
            df = pd.read_table(f)
        if args.filter_alfreq:
            df = df[df['A1FREQ'] > 0.01]  
        if args.quick_merge_mode == 3:
            df = df[df.LOG10P > args.ignore_threshold]
        # if args.filter_toploci_mode: #This is incorrect in my opinion, because non-toploci SNPs are also important for the plots
        #     top_loci = pd.read_table(g.replace(os.path.basename(g), f"toploci/toploci/{os.path.basename(g).split('.')[0]}.toploci.tsv"))
        #     unique_SNPs_toploci = set(top_loci[top_loci[args.col_toploci] >= args.min_toploci]['SNP'])
        #     top_loci[top_loci[args.col_toploci] >= args.min_toploci]['SP2'].dropna().str.split(',').apply(lambda x: unique_SNPs_toploci.update(i.split('(')[0] for i in x))
        #     df = df[df['ID'].isin(unique_SNPs_toploci)]
    df["Latent"] = os.path.basename(g).split(".")[0]
    df['ID_Latent'] = df['ID'].astype(str) + ' (' + os.path.basename(g).split(".")[0] + ')'
    return df

if __name__ == "__main__":
    parser = getARGSParser()
    args, _ = parser.parse_known_args() 

    if args.parallel:
        if 'SLURM_JOB_ID' in os.environ:
            n_proc = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            n_proc = mp.cpu_count()
    else:
        n_proc = 1

    if args.filter_toploci_mode:
        if args.filter_toploci_mode == 1:
            args.col_toploci = "TOTAL"
        elif args.filter_toploci_mode == 2:
            args.col_toploci = "S0001"

    args.path2gwasout = f"{args.path2gwasout}/results/gwas"
    gwas_outs = glob(f"{args.path2gwasout}/*.gz")
    corrected_sig_level = args.sig_level/len(gwas_outs) # Bonferroni correction: adjust based on the number of GWASs

    if args.process_dir_tophits:
        top_hits = glob(f"{args.path2gwasout}/tophits/*.gz")
        args.process_dir_tophits = len(top_hits) != 0 #if there are no top hits, then we need to process the GWAS outputs directly.
        processed_tophits = {}
        for h in top_hits:
            with gzip.open(h, 'rt') as f:
                df = pd.read_table(f)
                if len(df) > 0: #there are hits
                    processed_tophits[os.path.basename(h).split(".")[0]] = df
    if not args.process_dir_tophits:
        n_proc_ = min(n_proc, len(gwas_outs))
        if n_proc_ > 1:            
            print(f"Using {n_proc_} processes to parallelise the processing/discovery of the hits.")
            manager = mp.Manager()
            processed_tophits = manager.dict()
            with mp.Pool(processes=n_proc_) as pool:
                func = partial(process_hits, args=args, processed_tophits=processed_tophits)
                pool.map(func, gwas_outs)
            processed_tophits = dict(processed_tophits)
        else:
            processed_tophits = {}
            for g in gwas_outs:
                process_hits(g, args=args, processed_tophits=processed_tophits)

    if args.plot_only_hits:
        gwas_outs = [
            g
            for g in gwas_outs
            if os.path.basename(g).split(".")[0] in processed_tophits
        ]

    if not bool(args.path4res):
        args.path4res = f"{os.path.dirname(args.path2gwasout)}/visB"

    usecols = ['ID','CHROM','GENPOS','LOG10P', 'A1FREQ']

    subdir = ""
    if not args.process_dir_tophits:
        subdir = "customTopHits"
        if args.filter_alfreq:
            subdir += "_filtAlFrq"
        if args.filter_toploci_mode:
            subdir += f"_filtTopLoci_{args.min_toploci}{args.col_toploci}"  
    if not args.plot_only_hits:
        subdir += "_pltAll" if bool(subdir) else "pltAll"
    if bool(subdir):
        args.path4res += f"/{subdir}"

    os.makedirs(args.path4res, exist_ok=True)
    if args.gene_plot:
        os.makedirs(f"{args.path4res}/genename", exist_ok=True)
    if args.chpos_plot:
        os.makedirs(f"{args.path4res}/chpos", exist_ok=True)
    if args.snp_plot:
        os.makedirs(f"{args.path4res}/snpID", exist_ok=True)

    np.save(f"{args.path4res}/processed_tophits.npy", processed_tophits)

    if args.stratify_qq:
        bins = list(map(float, args.qq_bins.split(",")))
        args.qq_bins = list(zip(bins, bins[1:]))
        args.colour_qq_bins = args.colour_qq_bins.split(",")

    args.colours = args.colours.split(',')
    args.marker_size = list(map(int, args.marker_size.split(",")))

    if args.indi_plot:
        n_proc_ = min(n_proc, len(gwas_outs))
        if n_proc_ > 1:            
            print(f"Using {n_proc_} processes to parallelise the plotting.")
            with mp.Pool(processes=n_proc_) as pool:
                func = partial(plot_item, args=args, corrected_sig_level=corrected_sig_level, usecols=usecols)  
                pool.map(func, gwas_outs)
        else:
            for g in gwas_outs:
                plot_item(g, args, corrected_sig_level, usecols)

    if args.combo_plot:
        print("Plotting the combined Manhattan-QQ plots...")
        n_proc_ = min(n_proc, len(gwas_outs))
        if n_proc_ > 1:  
            print(f"Using {n_proc_} processes to parallelise the reading of the dataframes to be merged for the combo plots.")
            with mp.Pool(processes=n_proc_) as pool:
                func = partial(fetch_DF_combomerge, args=args, usecols=usecols) 
                g_outs = pool.map(func, gwas_outs)
        else:   
            g_outs = [fetch_DF_combomerge(g, args, usecols) for g in gwas_outs]

        if args.quick_merge_mode == 1:
            merged_df = reduce(quick_combo_merge_max, g_outs)
        elif args.quick_merge_mode == 2:
            merged_df = reduce(partial(quick_combo_merge_sigbased, args.sig_level), g_outs)
        else:
            merged_df = pd.concat(g_outs, ignore_index=True)

        savetag = ""
        if not args.limited_col_merge:
            savetag += "_limColMrg"
        if args.quick_merge_mode == 1:
            savetag += "_qckMaxMrg"
        elif args.quick_merge_mode == 2:
            savetag += "_qckSigMrg"
        elif args.quick_merge_mode == 3:
            savetag += f"_qckMrgIgnrPLrThn{str(args.ignore_threshold)}"

        merged_df.to_pickle(f"{args.path4res}/mergedDF_allhits_regenie{savetag}.pkl")

        mergedstats = gl.Sumstats(merged_df, fmt="regenie", other=['Latent', 'ID_Latent'], verbose=False)
        del merged_df, g_outs

        gl.dump_pickle(mergedstats, f"{args.path4res}/mergedsumstats_allhits{savetag}.pkl", overwrite=True)

        if args.gene_plot:
            mergedstats.plot_mqq(
                mode="b", colors = args.colours, scaled=True, sig_level=args.sig_level, sig_line_color=args.colour_sigline,
                suggestive_sig_line=True, suggestive_sig_level=corrected_sig_level, 
                skip=0 if args.quick_merge_mode != 3 else args.ignore_threshold, expected_min_mlog10p=0 if args.quick_merge_mode != 3 else args.ignore_threshold,
                suggestive_sig_line_color=args.colour_bonfline, anno="GENENAME", build="19", marker_size=args.marker_size,
                stratified=args.stratify_qq, maf_bins=args.qq_bins, maf_bin_colors=args.colour_qq_bins,
                verbose=False, save=f"{args.path4res}/genename/combo{savetag}_mqq.{args.ext}", 
                save_args={"dpi":args.dpi, "facecolor":"white"}
            )
            plt.close()

        if args.chpos_plot:
            mergedstats.plot_mqq(
                mode="b", colors = args.colours, scaled=True, sig_level=args.sig_level, sig_line_color=args.colour_sigline,
                suggestive_sig_line=True, suggestive_sig_level=corrected_sig_level, 
                skip=0 if args.quick_merge_mode != 3 else args.ignore_threshold, expected_min_mlog10p=0 if args.quick_merge_mode != 3 else args.ignore_threshold,
                suggestive_sig_line_color=args.colour_bonfline, anno=True, marker_size=args.marker_size,
                stratified=args.stratify_qq, maf_bins=args.qq_bins, maf_bin_colors=args.colour_qq_bins,
                verbose=False, save=f"{args.path4res}/chpos/combo{savetag}_mqq.{args.ext}", 
                save_args={"dpi":args.dpi, "facecolor":"white"}
            )
            plt.close()

        if args.snp_plot:
            mergedstats.plot_mqq(
                mode="b", colors = args.colours, scaled=True, sig_level=args.sig_level, sig_line_color=args.colour_sigline,
                suggestive_sig_line=True, suggestive_sig_level=corrected_sig_level, 
                skip=0 if args.quick_merge_mode != 3 else args.ignore_threshold, expected_min_mlog10p=0 if args.quick_merge_mode != 3 else args.ignore_threshold,
                suggestive_sig_line_color=args.colour_bonfline, anno="SNPID", marker_size=args.marker_size, 
                stratified=args.stratify_qq, maf_bins=args.qq_bins, maf_bin_colors=args.colour_qq_bins,
                verbose=False, save=f"{args.path4res}/snpID/combo{savetag}_mqq.{args.ext}", 
                save_args={"dpi":args.dpi, "facecolor":"white"}
            )
            plt.close()

        mergedstats.plot_mqq(
            mode="b", colors = args.colours, scaled=True, sig_level=args.sig_level, sig_line_color=args.colour_sigline,
            suggestive_sig_line=True, suggestive_sig_level=corrected_sig_level, 
                skip=0 if args.quick_merge_mode != 3 else args.ignore_threshold, expected_min_mlog10p=0 if args.quick_merge_mode != 3 else args.ignore_threshold,
            suggestive_sig_line_color=args.colour_bonfline, anno="ID_Latent", marker_size=args.marker_size, 
            stratified=args.stratify_qq, maf_bins=args.qq_bins, maf_bin_colors=args.colour_qq_bins,
            verbose=False, save=f"{args.path4res}/idlatent_combo{savetag}_mqq.{args.ext}", 
            save_args={"dpi":args.dpi, "facecolor":"white"}
        )
        plt.close()