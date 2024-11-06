import h5py
import argparse
import os
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", help="path to the zip files", default=r"")
parser.add_argument("--out_path", help="path to store the HDF5 file", default=r"../toysets")
parser.add_argument("--n_subjects", type=int, help="path to store the HDF5 file", default=10)
args = parser.parse_args()

args.out_path = f"{args.out_path}/dummy_{os.path.basename(args.in_path)}"
os.makedirs(args.out_path, exist_ok=True)

with h5py.File(f"{args.in_path}/data.h5", "r") as h5_file:
    group_names = list(h5_file.keys())

    random_groups = random.sample(group_names, k=args.n_subjects)

    with h5py.File(f"{args.out_path}/data.h5", 'w') as subset_h5:
        for group_name in tqdm(random_groups):
            # new_group = subset_h5.create_group(group_name)
            original_group = h5_file[group_name]
            original_group.copy(source=original_group, dest=subset_h5, name=group_name)
