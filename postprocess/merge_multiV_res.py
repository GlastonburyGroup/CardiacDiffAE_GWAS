from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm

import h5py

from ..H5tools.utils import merge_hdf5

parser = argparse.ArgumentParser()
parser.add_argument("--res_root", help="path to the zip files", default=r"../Out/Results")

parser.add_argument("--existing_res_out", help="What's the version tag of the original DS", default="Output_fullDSV2")
parser.add_argument("--new_res_out", help="What's the version tag of the original DS", default="Output_fullV3OnlyDS")
parser.add_argument("--merged_out", help="What's the version tag of the original DS", default="Output_fullDSV3")

parser.add_argument('--force_process', action=argparse.BooleanOptionalAction, default=False, help="Process even if they exists already! Ideally, use it by specifying path till the trainID (i.e. only one training)")

args = parser.parse_args()


embs = glob(f"{args.res_root}/**/{args.new_res_out}/emb.h5", recursive=True)
print(f"Found {len(embs)} emb.h5 files for merging")

for emb in tqdm(embs):
    print(emb)

    our_dir = os.path.dirname(emb).replace(args.new_res_out, args.merged_out)

    if not args.force_process:
        if os.path.isfile(f"{our_dir}/DSmerged.txt"):
            print(f"WARNING: {our_dir}/DSmerged.txt already exists. Meaning this was already merged. Skipping it!")
            continue
        elif os.path.isfile(f"{our_dir}/emb.h5"):
            print(f"WARNING: {our_dir}/emb.h5 already exists, but {our_dir}/DSmerged.txt does not exist. Meaning this was directly inferred using the V2 dataset. Skipping it!")
            continue

    os.makedirs(our_dir, exist_ok=True)

    existing_dir = os.path.dirname(emb).replace(args.new_res_out, args.existing_res_out)

    #emb.h5
    if os.path.isfile(f"{existing_dir}/emb.h5"):
        try:
            merge_hdf5(orig_file=f"{existing_dir}/emb.h5", new_file=emb, dest_file=f"{our_dir}/emb.h5")
        except Exception as e:
            print(f"ERROR: in copying {existing_dir}/emb.h5 to {our_dir}/emb.h5: {e}")
    else:
        print(f"WARNING: {existing_dir}/emb.h5 does not exist, skipping it")

    #recon.h5
    if os.path.isfile(f"{existing_dir}/recon.h5") and os.path.isfile(emb.replace("emb.h5", "recon.h5")):
        try:
            merge_hdf5(orig_file=f"{existing_dir}/recon.h5", new_file=emb.replace("emb.h5", "recon.h5"), dest_file=f"{our_dir}/recon.h5")
        except Exception as e:
            print(f"ERROR: in copying {existing_dir}/recon.h5 to {our_dir}/recon.h5: {e}")
    else:
        print(f"WARNING: {existing_dir}/recon.h5 or {emb.replace('emb.h5', 'recon.h5')} does not exist, skipping it")

    #metrics.csv
    if os.path.isfile(f"{existing_dir}/metrics.csv") and os.path.isfile(emb.replace("emb.h5", "metrics.csv")):
        df_existing = pd.read_csv(f"{existing_dir}/metrics.csv")
        df_new = pd.read_csv(emb.replace("emb.h5", "metrics.csv"))
        df_out = pd.concat([df_existing, df_new])
        df_out.to_csv(f"{our_dir}/metrics.csv", index=False)
    else:
        print(f"WARNING: {existing_dir}/metrics.csv or {emb.replace('emb.h5', 'metrics.csv')} does not exist, skipping it")

    #metrics.pkl is useless, so we won't merge it
    
    #maybe add metrics_wSplits_consolidated.csv as well? For now, skipping it

    with open(f"{our_dir}/DSmerged.txt", "w") as f:
        f.write("Just to mark that this directory has been created by merging the two versions\n")
        f.write(f"Existing version: {existing_dir}\n")
        f.write(f"New version: {emb}\n")