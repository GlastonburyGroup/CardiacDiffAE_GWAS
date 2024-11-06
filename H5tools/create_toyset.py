"""
This script creates the HDF5 files from the zip files downloaded from UK Biobank.
This is a generic version, was used for the creation of the HDF5 files for the short and long axis heart images.
"""

import os
import h5py
import argparse
import random

random.seed(1701)

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", help="path to store the HDF5 file", default="data.h5")
parser.add_argument("--out_path", help="path to store the HDF5 file", default="toydata.h5")
parser.add_argument("--n_subjects", type=int, default=5)
args = parser.parse_args()

with h5py.File(args.in_path, 'r') as f:    
    keys = list(f.keys())
    random.shuffle(keys)
    keys = keys[:args.n_subjects]
    print(keys)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    with h5py.File(args.out_path, 'w') as f_out:
        for key in keys:
            print(f"Copying {key}")
            f.copy(key, f_out)            

print("Done!")