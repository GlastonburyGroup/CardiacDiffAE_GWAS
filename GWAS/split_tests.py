import argparse
from glob import glob
from tqdm import tqdm
import gzip
import pandas as pd
import os
import multiprocessing as mp
from functools import partial

def calculate_maf(a1freq):
    a2freq = 1 - a1freq
    return min(a1freq, a2freq)

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path2gwasout', type=str, default="", help='Location where the GWAS outputs are storred [The main folder, containing a subfolder results, containing subfolder defined by res_type param]')
    parser.add_argument('--res_type', type=str, default="gwas", help='Set it to the type of result, gwas (default) or ewas or burden')
    parser.add_argument('--filter_maf', action=argparse.BooleanOptionalAction, default=True, help="Whether to filter the GWAS tophits based on the allele frequency (>0.01). [Only if process_dir_tophits is False]")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=False, help="If set to True and there are multiple GWAS outs to plot, the plotting will be parallelised.")
    
    return parser

def split_tests(g, args):
    with gzip.open(g, 'rt') as f:
        df = pd.read_table(f)
    if args.filter_maf:
        df['MAF'] = df['A1FREQ'].apply(calculate_maf) 
        df = df[df['MAF'] > 0.01]
    for test in df["TEST"].unique():
        os.makedirs(f"{args.path2gwasout}/results/{args.res_type}_{test}", exist_ok=True)
        df_filtered = df[df["TEST"] == test]
        df_filtered.to_csv(f"{args.path2gwasout}/results/{args.res_type}_{test}/{os.path.basename(g)}", sep="\t", index=False)

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

    args.path2gwasout = f"{args.path2gwasout}/results/{args.res_type}"
    gwas_outs = glob(f"{args.path2gwasout}/*.gz")

    n_proc_ = min(n_proc, len(gwas_outs))
    if n_proc_ > 1:            
        print(f"Using {n_proc_} processes to parallelise the processing/discovery of the hits.")
        manager = mp.Manager()
        processed_tophits = manager.dict()
        with mp.Pool(processes=n_proc_) as pool:
            func = partial(split_tests, args=args)
            pool.map(func, gwas_outs)
        processed_tophits = dict(processed_tophits)
    else:
        processed_tophits = {}
        for g in tqdm(gwas_outs):
            split_tests(g, args=args)