from glob import glob
import h5py
import argparse
import os
import shutil

import h5py

def copy_hdf5(source_file, dest_file):
    """
    Copy the content of source_file to dest_file using h5py.
    
    :param source_file: path to the source HDF5 file.
    :param dest_file: path to the destination HDF5 file.
    """
    
    with h5py.File(source_file, 'r') as src:
        with h5py.File(dest_file, 'a') as dest:
            for name, _ in src.items():
                new_name = name
                while new_name in dest:
                    print(f"Existing item encountered! Renaming {new_name} to {new_name}_new")
                    new_name += '_new'
                src.copy(new_name, dest)

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
parser.add_argument("--new_root", help="path to the zip files", default=r"../ukbbH5s/_tmp_newV3")
parser.add_argument("--original_root", help="path to store the HDF5 file", default=r"../ukbbH5s")
parser.add_argument("--out_root", help="path to store the HDF5 file", default=r"../ukbbH5s")

parser.add_argument("--skips", default=r"F20208", help="Folders (coma-seperated) to skip (from the new root)")

parser.add_argument("--originalV", help="What's the version tag of the original DS", default="v2")
parser.add_argument("--newV", help="What's the version tag of newly downloaded DS", default="OnlyV3")
parser.add_argument("--outV", help="What's the version of the output DS", default="v3")

args = parser.parse_args()

if args.newV:
    new_dirs = glob(f"{args.new_root}/*_H5{args.newV}*")
else:
    new_dirs = glob(f"{args.new_root}/*_H5*")

if bool(args.skips):
    skips = args.skips.split(",")
    new_dirs = [x for x in new_dirs if all(skip not in x for skip in skips)]

for new_dir in new_dirs:
    print(new_dir)
    original_dir = new_dir.replace(args.new_root, args.original_root).replace(args.newV, args.originalV)
    out_dir = new_dir.replace(args.new_root, args.out_root).replace(args.newV, args.outV)

    original_H5s = glob(f"{original_dir}/*.h5")
    for origH5 in original_H5s:
        os.makedirs(out_dir, exist_ok=True)
        try:
            shutil.copyfile(origH5, f"{out_dir}/{os.path.basename(origH5)}")
            merge_hdf5(source_file=f"{new_dir}/{os.path.basename(origH5)}", dest_file=f"{out_dir}/{os.path.basename(origH5)}")
        except Exception as e:
            print(f"Error in copying {origH5}. The error is {e}")