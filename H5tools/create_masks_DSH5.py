"""
This script creates the HDF5 files from the zip files downloaded from UK Biobank.
This is a generic version, was used for the creation of the HDF5 files for the short and long axis heart images.
"""

import numpy as np
import json
from glob import glob
from zipfile import ZipFile
import h5py
import argparse
import pandas as pd
import os
from tqdm import tqdm
import SimpleITK as sitk
import tempfile
import logging
from tricorder.mri.data.dicom import ReadSeries
import json
from collections import defaultdict
import scipy.ndimage as ndimg

from skimage import measure
from skimage import morphology
from skimage import segmentation
from skimage import filters


# read the first command line argument and set the detault to "C:\Users\Public\Documents\HDF5\*.zip" using argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", help="path to store the HDF5 file", default=r"../datasets/ukbbH5s/_tmp_newV3/F20208_Long_axis_heart_images_DICOM_H5")
parser.add_argument("--out_file", help="output file name (default: meta_mask.h5), to be storred inside the <in_path>", default="meta_mask.h5")
parser.add_argument("--mode", type=int, help="0: Create heart masks (20208)", default=0)
args = parser.parse_args()

mask_file = h5py.File(f"{args.in_path}/{args.out_file}", 'w')

def create_masks(name, obj, mode):
    if isinstance(obj, h5py.Dataset) and not name.startswith('meta_mask'):
        
        match mode:
            case 0: #heart mask
                vari = np.var(obj, axis=1, keepdims=True)

                binary_mask = ndimg.binary_closing(vari>(vari.mean()*0.5), structure=np.ones((1,1,1,10,10))).astype(np.float64)
                binary_mask = ndimg.binary_opening(binary_mask, structure=np.ones((1,1,1,5,5))).astype(np.float64)
                binary_mask = ndimg.median_filter(binary_mask, size=5)

                label_im, nb_labels = ndimg.label(binary_mask)
                sizes = ndimg.sum(binary_mask, label_im, range(nb_labels + 1))
                mask = (sizes == max(sizes))

                binary_mask = mask[label_im].astype(np.int8)
            case 1: #Pancreas 20259
                mag = abs(obj)
                mask = mag > np.percentile(mag, 50)

                labels = measure.label(mask)
                props = measure.regionprops(labels)
                areas = [prop.area for prop in props]
                largest_label = props[areas.index(max(areas))].label
                largest_component = labels == largest_label

                mask_processed = morphology.remove_small_objects(largest_component, min_size=64)

                mask_processed = morphology.binary_closing(mask_processed, morphology.disk(4))
                binary_mask = morphology.binary_opening(mask_processed, morphology.disk(4))

        path_parts = name.split('/')
        current_mskgroup = mask_file
        for part in path_parts[:-1]:
            if part not in current_mskgroup:
                current_mskgroup.create_group(part)
            current_mskgroup = current_mskgroup[part]

        dset = current_mskgroup.create_dataset(path_parts[-1], data=binary_mask)
        dset.attrs["min_val"] = 0
        dset.attrs["max_val"] = 1
        dset.attrs["seriesID"] = obj.attrs['seriesID']

with h5py.File(args.in_path if args.in_path.endswith('.h5') else f"{args.in_path}/data.h5", 'r') as f:
    f.visititems(lambda name, obj: create_masks(name, obj, args.mode))

print("Done!")