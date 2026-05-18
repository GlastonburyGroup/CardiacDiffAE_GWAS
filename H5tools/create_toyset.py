"""
This script creates the HDF5 files from the zip files downloaded from UK Biobank.
This is a generic version, was used for the creation of the HDF5 files for the short and long axis heart images.
"""

import os
import sys
sys.path.insert(0, os.getcwd()) #to handle the sub-foldered structure of the tricoder

import h5py
import argparse
import random
from preprocess.dsSpliter import split_dataset

random.seed(1701)

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", help="path to store the HDF5 file", default=r"/scratch/glastonbury/datasets/ukbbH5s/F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3/data.h5")
parser.add_argument("--out_path", help="path to store the HDF5 file", default=r"/group/glastonbury/soumick/toysets/dummy_F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3/data.h5")
parser.add_argument("--n_subjects", type=int, default=5)
parser.add_argument("--create_folds", action=argparse.BooleanOptionalAction, help="Whether to create fold CSVs directly after creating the dummy H5.", default=True)
parser.add_argument('--ds_names_present', type=str, action="store", default="primary_0", help="if it is not empty (coma-separated list), then only patients with this dataset name will be used")
args = parser.parse_args()

with h5py.File(args.in_path, 'r') as f: 
    if args.ds_names_present != "":  
        ds_names_present_list = args.ds_names_present.split(',') if ',' in args.ds_names_present else args.ds_names_present.split('OR')
        IDs = []      
        def dsFind(name, obj):
            if isinstance(obj, h5py.Dataset):
                for dsname in ds_names_present_list:
                    if dsname in name:
                        IDs.append(name.split('/')[0])
                        break
        f.visititems(dsFind)
        keys = list(set(IDs))
    else:
        keys = list(f.keys())

    random.shuffle(keys)
    keys = keys[:args.n_subjects]
    print(keys)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    with h5py.File(args.out_path, 'w') as f_out:
        for key in keys:
            print(f"Copying {key}")
            f.copy(key, f_out)            

print("Dummy H5 created!")

if args.create_folds:
    print("Creating folds...")
    split_dataset(args.out_path, seed=1701, per_dataset=1, per_train=0.6, per_val=0.2, per_test=0.2, n_folds=5, patient_n_sessions=-1, ds_names_present=args.ds_names_present)
    print("Folds created!")