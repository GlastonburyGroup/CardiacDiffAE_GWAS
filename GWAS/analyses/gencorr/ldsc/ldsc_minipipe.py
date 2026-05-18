import argparse
import os
from glob import glob
from copy import deepcopy
from tqdm import tqdm
import itertools
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

from munge_sumstats import munge_sumstats, parser as munge_parser
from ldsc import main as get_ldsc, parser as ldsc_parser

def munge_sumstats_parallel(p_sumstat, args):
    phenotype = os.path.basename(p_sumstat).split(".")[0]
    _tmp_args = deepcopy(args)
    _tmp_args.sumstats = p_sumstat
    _tmp_args.out = os.path.join(args.munge_out_root, phenotype)
    munge_sumstats(_tmp_args)
    return phenotype, _tmp_args.out

def get_ldsc_parallel(combo, args, paths_mungedstats):
    _tmp_args = deepcopy(args)
    _tmp_args.rg = f"{paths_mungedstats[combo[0]]}.sumstats.gz,{paths_mungedstats[combo[1]]}.sumstats.gz"
    _tmp_args.out = os.path.join(args.ldsc_out_root, f"{combo[0]}_{combo[1]}")
    get_ldsc(_tmp_args)

def getARGSParser(parents=None):     
    parser = argparse.ArgumentParser()

    parser.add_argument('--path2gwasout', type=str, default="/group/glastonbury/sara/gwas/WBRIT_time2slc_Msk_V2_3D_L1_128FVAE_fullDS/results/gwas", help='Location where the GWAS outputs are storred')
    parser.add_argument('--path4res', type=str, default="", help='Path where the genetic correlation outputs will be storred [If blank, it will be storred inside the LDSC folder present in the parent folder of path2gwasout]')
    parser.add_argument('--path2LDScoresWeights', type=str, default="/group/glastonbury/sara/postgwas/gencorr/input/eur_w_ld_chr/", help='Path to the folder containing LD scores and weights')
    
    parser.add_argument('--n_workers', type=int, default=10, help='Number of workers to use for parallelisation. Set it to -1 for using all available CPUs')

    parser.add_argument('--munge_only', action=argparse.BooleanOptionalAction, default=True, help="If only munging step (Step 0) is to be performed")

    #Param for munging
    parser.add_argument('--snplist', type=str, default="/group/glastonbury/sara/postgwas/gencorr/input/w_hm3.snplist")

    for parent in parents:
        for action in parent._actions:
            if action.dest not in ['help', 'out']:
                parser._add_action(action)

    return parser

def define_defaults(args):
    #Munge
    args.a1 = "ALLELE0"
    args.a2 = "ALLELE1"
    args.signed_sumstats = "BETA,0"
    args.raw_regenie = True
    args.merge_alleles = args.snplist

    #LDSC
    args.ref_ld_chr = (args.path2LDScoresWeights if args.path2LDScoresWeights[-1] == "/" else f"{args.path2LDScoresWeights}/")
    args.w_ld_chr = (args.path2LDScoresWeights if args.path2LDScoresWeights[-1] == "/" else f"{args.path2LDScoresWeights}/")

    return args

if __name__ == "__main__":
    parser = getARGSParser(parents=[munge_parser, ldsc_parser])
    args, _ = parser.parse_known_args()
    args = define_defaults(args)

    if not bool(args.path4res):
        args.path4res = f"{os.path.dirname(args.path2gwasout)}/LDSC"

    args.munge_out_root = os.path.join(args.path4res, "Step0_Munge")
    os.makedirs(args.munge_out_root, exist_ok=True)

    if not args.munge_only:
        args.ldsc_out_root = os.path.join(args.path4res, "Step1_LDSC")
        os.makedirs(args.ldsc_out_root, exist_ok=True)

    paths_sumstats = glob(os.path.join(args.path2gwasout, "*.regenie.gz"))
    
    if args.n_workers == -1:
        args.n_workers = os.cpu_count()

    #Munge the sumstats
    print("Munging sumstats...")
    paths_mungedstats = {}
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [executor.submit(munge_sumstats_parallel, p_sumstat, args) for p_sumstat in paths_sumstats]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(paths_sumstats)):
            phenotype, out_path = future.result()
            paths_mungedstats[phenotype] = out_path

    #Run LDSC
    if not args.munge_only:
        print("Running LDSC...")
        combos = list(itertools.combinations(list(paths_mungedstats.keys()), 2))
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [executor.submit(get_ldsc_parallel, combo, args, paths_mungedstats) for combo in combos]

            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(combos)):
                pass

print("Live long and prosper!")