from glob import glob
import h5py
import argparse
import os
import shutil
import pandas as pd
from tqdm import tqdm

import h5py

def merge_hdf5(source_file, dest_file):
    """
    Merge the content of source_file into dest_file using h5py.
    
    :param source_file: path to the source HDF5 file.
    :param dest_file: path to the destination HDF5 file.
    """
    
    with h5py.File(source_file, 'r') as src:
        with h5py.File(dest_file, 'a') as dest:
            _merge_group(src, dest)

def _merge_group(src_group, dest_group):
    """Recursively merge groups from src_group into dest_group."""
    for name, item in src_group.items():
        if name in dest_group:
            # If both are groups, merge their content
            if isinstance(item, h5py.Group) and isinstance(dest_group[name], h5py.Group):
                _merge_group(item, dest_group[name])
            elif isinstance(item, h5py.Dataset) and isinstance(dest_group[name], h5py.Dataset):
                print(f"Existing item encountered! Renaming {name} to {name}_new")
                dest_group[f"{name}_new"] = item
        else:
            src_group.copy(name, dest_group)

parser = argparse.ArgumentParser()
parser.add_argument("--res_root", help="path to the zip files", default=r"/project/ukbblatent/Out/Results")

parser.add_argument("--existing_res_out", help="What's the version tag of the original DS", default="Output_fullDS")
parser.add_argument("--new_res_out", help="What's the version tag of the original DS", default="Output_fullV2OnlyDS")
parser.add_argument("--merged_out", help="What's the version tag of the original DS", default="Output_fullDSV2")

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
            shutil.copyfile(f"{existing_dir}/emb.h5", f"{our_dir}/emb.h5")
            merge_hdf5(source_file=emb, dest_file=f"{our_dir}/emb.h5")
        except Exception as e:
            print(f"ERROR: in copying {existing_dir}/emb.h5 to {our_dir}/emb.h5: {e}")
    else:
        print(f"WARNING: {existing_dir}/emb.h5 does not exist, skipping it")

    #recon.h5
    if os.path.isfile(f"{existing_dir}/recon.h5") and os.path.isfile(emb.replace("emb.h5", "recon.h5")):
        try:
            shutil.copyfile(f"{existing_dir}/recon.h5", f"{our_dir}/recon.h5")
            merge_hdf5(source_file=emb.replace("emb.h5", "recon.h5"), dest_file=f"{our_dir}/recon.h5")
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