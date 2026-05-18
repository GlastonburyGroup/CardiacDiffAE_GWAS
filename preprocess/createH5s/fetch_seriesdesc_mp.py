"""
This script creates the HDF5 files from the zip files downloaded from UK Biobank.
This is a generic version, was used for the creation of the HDF5 files for the short and long axis heart images.
"""


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
import sys
import contextlib
import yaml
import numpy as np
import collections 
from collections import Counter
import multiprocessing

def determine_orientation(image_ori):
    if type(image_ori) is str:
        image_ori = image_ori.split("\\")
        image_ori = [float(x) for x in image_ori]
    image_y = np.array([image_ori[0], image_ori[1], image_ori[2]])
    image_x = np.array([image_ori[3], image_ori[4], image_ori[5]])
    image_z = np.cross(image_x, image_y)
    abs_image_z = abs(image_z)
    main_index = list(abs_image_z).index(max(abs_image_z))
    if main_index == 0:
        return "sagittal"
    elif main_index == 1:
        return "coronal"
    else:
        return "transverse"

# read the first command line argument and set the detault to "C:\Users\Public\Documents\HDF5\*.zip" using argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", help="path to the zip files", default=r"/group/glastonbury/soumick/dataset/UKBBDownloads/ukb673493_fourth_basket/bulkfiles")
# parser.add_argument("--in_path", help="path to the zip files", default=r"/project/ukbblatent/toysets/dummy_F20208_Long_axis_heart_images_DICOM")
parser.add_argument("--use_SimpleITK", action=argparse.BooleanOptionalAction, help="whether to use SimpleITK or PyDicom", default=True)
#If we are processing the directory of unsorted bulk files
parser.add_argument("--dir_unsorted", action=argparse.BooleanOptionalAction, help="Whether processing the unsorted directory of bulk files (freshly downloaded)", default=True)
parser.add_argument("--fID", help="fieldID of the bulk files to process", default="20208")

args = parser.parse_args()

print("Multiprocessing Version")
print(args.in_path)

# read the zip files
if args.dir_unsorted:
    zip_files = glob(f"{args.in_path}/*_{args.fID}_*.zip")
else:
    zip_files = glob(f"{args.in_path}/*.zip")
    args.fDirName = os.path.basename(args.in_path)

print(f"Found {len(zip_files)} zip files")

def process_zip_file(zip_file):
    n_series_local = []
    seriesDesc_local = []
    orientation_local = []
    try:
        # read the zip file
        with ZipFile(zip_file, "r") as zip_ref:
            # extract the zip file into Temporary Directory
            with tempfile.TemporaryDirectory(prefix="createH5_MRI_") as tmp_dir:
                zip_ref.extractall(tmp_dir)
                
                # read manifest.cvs file from inside the zip
                df = pd.read_csv(glob(f"{tmp_dir}/manifest.*")[0], on_bad_lines='skip') # sometimes the manifest file is manifest.csv and sometimes manifest.cvs
                n_series = len(df.seriesid.unique())
                n_series_local.append(n_series)
                
                seriesIDs = sorted(df.seriesid.unique())
                for i, seriesID in enumerate(seriesIDs):
                    series, seriesMeta = ReadSeries(tmp_dir, return_meta=True, taginits2ignore=["0029"], series_ids=seriesID, series2array=False)
                    if len(series) == 0:                            
                        logging.error(f"Dirty DICOM: In {zip_file}, for seriesID: {seriesID} (the {i+1}th series out of {n_series} series), has an issue with the DICOM files. It will be skipped.")
                        continue
                    seriesDesc = df[df.seriesid==seriesID]['series discription'].unique()
                    if len(seriesDesc) > 1:
                        logging.warning(f"Warning: More than one series discription found in {zip_file}, for seriesID: {seriesID}")
                    seriesDesc_local.append(seriesDesc[0])
                    orientation_local.append(determine_orientation(seriesMeta[0]['0020|0037']))
    except:
        pass
    if len(orientation_local) > 0:
        return n_series_local, seriesDesc_local, orientation_local
    else:
        return None

pool = multiprocessing.Pool(processes=4)
results = pool.map(process_zip_file, zip_files)

n_series_master = []
seriesDesc_master = []
orientation_master = []

for r in results:
    if r is not None:
        n_series_master += r[0]
        seriesDesc_master += r[1]
        orientation_master += r[2] 

print("\n-----------------------\n")
print(args.in_path)
print("\n-----------------------\n")

item_counts = Counter(seriesDesc_master)
unique_items = list(item_counts.keys())
print("seriesDesc Unique items:", unique_items)
print("seriesDesc Item counts:", item_counts)

item_counts = Counter(orientation_master)
unique_items = list(item_counts.keys())
print("orientation Unique items:", unique_items)
print("orientation Item counts:", item_counts)

item_counts = Counter(n_series_master)
unique_items = list(item_counts.keys())
print("n_series Unique items:", unique_items)
print("n_series Item counts:", item_counts)