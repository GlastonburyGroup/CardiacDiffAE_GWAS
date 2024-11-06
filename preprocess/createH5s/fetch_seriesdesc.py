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

def readSingleValMeta(df, field):
    metas = df[field].unique()
    if len(metas) > 1:
        logging.warning(f"Warning: More than one {field} found in {zip_file}")
    return metas[0]

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
parser.add_argument("--in_path", help="path to the zip files", default=r"")
parser.add_argument("--out_path", help="path to store the HDF5 file", default=r"")
parser.add_argument("--use_SimpleITK", action=argparse.BooleanOptionalAction, help="whether to use SimpleITK or PyDicom", default=True)
#If we are processing the directory of unsorted bulk files
parser.add_argument("--dir_unsorted", action=argparse.BooleanOptionalAction, help="Whether processing the unsorted directory of bulk files (freshly downloaded)", default=True)
parser.add_argument("--fID", help="fieldID of the bulk files to process", default="20208")

args = parser.parse_args()

print(args.in_path)

# read the zip files
if args.dir_unsorted:
    zip_files = glob(f"{args.in_path}/*_{args.fID}_*.zip")
else:
    zip_files = glob(f"{args.in_path}/*.zip")
    args.fDirName = os.path.basename(args.in_path)

print(f"Found {len(zip_files)} zip files")

os.makedirs(args.out_path, exist_ok=True)

logging.basicConfig(filename=f"{args.out_path}/log.txt", level=logging.DEBUG)

seriesDesc_master = []
orientation_master = []
n_series_master = []
# loop over the zip files
for zip_file in tqdm(zip_files):
    try:
        # read the zip file
        with ZipFile(zip_file, "r") as zip_ref:
            # extract the zip file into Temporary Directory
            with tempfile.TemporaryDirectory(prefix="createH5_MRI_") as tmp_dir:
                zip_ref.extractall(tmp_dir)

                fileID = os.path.basename(zip_file).replace(".zip","")
                patientID, fieldID, instanceID, unknownID = fileID.split("_")
                instanceID = f"{instanceID}_{unknownID}" #as we don't know what is the meaning of that unknownID, we will just add it to the instanceID
                
                # read manifest.cvs file from inside the zip
                df = pd.read_csv(glob(f"{tmp_dir}/manifest.*")[0], on_bad_lines='skip') # sometimes the manifest file is manifest.csv and sometimes manifest.cvs
                n_series = len(df.seriesid.unique())
                n_series_master.append(n_series)
                
                seriesIDs = sorted(df.seriesid.unique())
                series_dataset = collections.defaultdict(dict)
                for i, seriesID in enumerate(seriesIDs):
                    series, seriesMeta = ReadSeries(tmp_dir, return_meta=True, taginits2ignore=["0029"], series_ids=seriesID, series2array=False)
                    if len(series) == 0:                            
                        logging.error(f"Dirty DICOM: In {zip_file}, for seriesID: {seriesID} (the {i+1}th series out of {n_series} series), has an issue with the DICOM files. It will be skipped.")
                        continue
                    seriesDesc = df[df.seriesid==seriesID]['series discription'].unique()
                    if len(seriesDesc) > 1:
                        logging.warning(f"Warning: More than one series discription found in {zip_file}, for seriesID: {seriesID}")
                    seriesDesc_master.append(seriesDesc[0])
                    orientation_master.append(determine_orientation(seriesMeta[0]['0020|0037']))
    except:
        pass

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