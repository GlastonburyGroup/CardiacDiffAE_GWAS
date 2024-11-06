import shutil
import h5py
import os

def merge_hdf5(orig_file, new_file, dest_file=""):
    """
    Merge the content of source_file into dest_file using h5py.
    
    :param orig_file: path to the original HDF5 file, that will be first copied to the destination file.
    :param new_file: path to the source HDF5 file - content of which will be appended to the destination file.
    :param dest_file: path to the destination HDF5 file. Can be left blank or set to None, then it will be set to orig_file.
    
    If dest_file is not set, and the orig_file does not exist, then the source_file will be copied to orig_file.
    If dest_file is not set, and the orig_file exists, then the source_file will be copied to orig_file and then new_file will be appended to orig_file.
    If dest_file is set, then the orig_file will be copied to dest_file and then new_file will be appended to dest_file.
    """
    if not bool(dest_file) and not os.path.isfile(orig_file):
        shutil.copyfile(new_file, orig_file)
    else:
        if bool(dest_file):
            shutil.copyfile(orig_file, dest_file)
        else:
            dest_file = orig_file
    
        with h5py.File(new_file, 'r') as src:
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