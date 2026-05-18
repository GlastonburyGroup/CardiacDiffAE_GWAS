import sys
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

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path1', type=str, default="", help='Location where the GWAS outputs are storred')
    parser.add_argument('--path2', type=str, default="", help='Location where the GWAS outputs are storred')
    parser.add_argument('--titles', type=str, default="", help='Location where the GWAS outputs are storred')
    
    parser.add_argument('--path4res', type=str, default="", help='Path where the genetic correlation outputs will be storred')
    parser.add_argument('--restag', type=str, default="", help='Tag for the output files')
    
    parser.add_argument("--sig_level", type=float, default=5e-8, help="Significance level.")
    parser.add_argument("--n4bonferroni", type=int, default=128, help="Number of GWASs for Bonferroni correction.")
    parser.add_argument("--ignore_threshold", default=2, type=int, help="If quick merge mode is 3, then varaints with LOG10P less than this value will be ignored.")
    
    parser.add_argument("--colours1", type=str, default= "#597FBD,#74BAD3", help="Two colours for the manhatten plot.")
    parser.add_argument("--colours2", type=str, default= "#597FBD,#74BAD3", help="Two colours for the manhatten plot.")
    parser.add_argument("--colour_sigline", default="blue", help="Colour of the significance line.")
    parser.add_argument("--colour_bonfline", default="red", help="Colour of the Bonferroni line.")
    parser.add_argument("--marker_size", type=str, default= "5,10", help="Marker size.")
    parser.add_argument("--titles_pad", type=str, default= "0.0,0.3", help="Padding for the subplot titles.")

    parser.add_argument("--dpi", type=int, default=300, help="DPI of the plots.")
    parser.add_argument("--ext", default="png", help="Saving extension.")

    parser.add_argument("--gene_plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to create plots with gene names.")
    parser.add_argument("--chpos_plot", action=argparse.BooleanOptionalAction, default=False, help="Whether to create plots with chromosome positions.")
    parser.add_argument("--snp_plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to create plots with the SNP IDs.")
    parser.add_argument("--idlatent_plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to create plots with the SNP IDs.")
    
    return parser

if __name__ == "__main__":
    parser = getARGSParser()
    args, _ = parser.parse_known_args() 
    
    args.path4res = f"{args.path4res}/{args.restag}"
    os.makedirs(args.path4res, exist_ok=True)

    args.colours1 = args.colours1.split(',')
    args.colours2 = args.colours2.split(',')
    args.titles = args.titles.split(',')
    args.marker_size = list(map(int, args.marker_size.split(",")))
    args.titles_pad = list(map(float, args.titles_pad.split(",")))

    df1 = pd.read_pickle(args.path1)
    df2 = pd.read_pickle(args.path2)
    corrected_sig_level = args.sig_level/args.n4bonferroni # Bonferroni correction: adjust based on the number of GWASs

    if args.gene_plot:
        gl.plot_miami(path1=df1, path2=df2, titles=args.titles,
                        cols1=['CHROM','GENPOS','LOG10P'], 
                        cols2=['CHROM','GENPOS','LOG10P'], 
                        colors1 = args.colours1, colors2 = args.colours2,
                        scaled=True, sig_level=args.sig_level, sig_line_color=args.colour_sigline,
                        additional_line=[corrected_sig_level], additional_line_color=[args.colour_bonfline],
                        skip=args.ignore_threshold, marker_size=args.marker_size, titles_pad=args.titles_pad,
                        anno="GENENAME", build="19", verbose=False, 
                        save=f"{args.path4res}/genename_miami.{args.ext}", 
                        save_args={"dpi":args.dpi, "facecolor":"white"}
                    )
        plt.close()

    if args.chpos_plot:
        gl.plot_miami(path1=df1, path2=df2, titles=args.titles,
                        cols1=['CHROM','GENPOS','LOG10P'], 
                        cols2=['CHROM','GENPOS','LOG10P'], 
                        colors1 = args.colours1, colors2 = args.colours2,
                        scaled=True, sig_level=args.sig_level, sig_line_color=args.colour_sigline,
                        additional_line=[corrected_sig_level], additional_line_color=[args.colour_bonfline],
                        skip=args.ignore_threshold, marker_size=args.marker_size, titles_pad=args.titles_pad,
                        anno=True, verbose=False, 
                        save=f"{args.path4res}/chpos_miami.{args.ext}", 
                        save_args={"dpi":args.dpi, "facecolor":"white"}
                    )
        plt.close()

    if args.snp_plot:
        gl.plot_miami(path1=df1, path2=df2, titles=args.titles,
                        cols1=['CHROM','GENPOS','LOG10P', 'ID'], 
                        cols2=['CHROM','GENPOS','LOG10P', 'ID'], 
                        colors1 = args.colours1, colors2 = args.colours2,
                        scaled=True, sig_level=args.sig_level, sig_line_color=args.colour_sigline,
                        additional_line=[corrected_sig_level], additional_line_color=[args.colour_bonfline],
                        skip=args.ignore_threshold, marker_size=args.marker_size, titles_pad=args.titles_pad,
                        anno="ID", verbose=False, 
                        save=f"{args.path4res}/snpID_miami.{args.ext}", 
                        save_args={"dpi":args.dpi, "facecolor":"white"}
                    )
        plt.close()

    if args.idlatent_plot:
        gl.plot_miami(path1=df1, path2=df2, titles=args.titles,
                        cols1=['CHROM','GENPOS','LOG10P', 'ID_Latent'], 
                        cols2=['CHROM','GENPOS','LOG10P', 'ID_Latent'], 
                        colors1 = args.colours1, colors2 = args.colours2,
                        scaled=True, sig_level=args.sig_level, sig_line_color=args.colour_sigline,
                        additional_line=[corrected_sig_level], additional_line_color=[args.colour_bonfline],
                        skip=args.ignore_threshold, marker_size=args.marker_size, titles_pad=args.titles_pad,
                        anno="ID_Latent", verbose=False, 
                        save=f"{args.path4res}/idlatent_miami.{args.ext}", 
                        save_args={"dpi":args.dpi, "facecolor":"white"}
                    )
        plt.close()