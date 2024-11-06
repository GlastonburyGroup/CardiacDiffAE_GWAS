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

sys.path.insert(0, os.getcwd())

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
parser.add_argument("--in_path", help="path to the zip files", default=r"../basket_bulk/bulkfiles")
parser.add_argument("--out_path", help="path to store the HDF5 file", default=r"../datasets/ukbbH5s/_tmp_newV3")
parser.add_argument("--use_SimpleITK", action=argparse.BooleanOptionalAction, help="whether to use SimpleITK or PyDicom", default=True)

#If we are processing the directory of unsorted bulk files
parser.add_argument("--dir_unsorted", action=argparse.BooleanOptionalAction, help="Whether processing the unsorted directory of bulk files (freshly downloaded)", default=True)
parser.add_argument("--fID", help="fieldID of the bulk files to process", default="20208")
parser.add_argument("--fDirName", help="Name of the directory for the particular filed", default="F20208_Long_axis_heart_images_DICOM")

args = parser.parse_args()

# read the zip files
if args.dir_unsorted:
    zip_files = glob(f"{args.in_path}/*_{args.fID}_*.zip")
else:
    zip_files = glob(f"{args.in_path}/*.zip")
    args.fDirName = os.path.basename(args.in_path)

print(f"Found {len(zip_files)} zip files")

args.out_path = f"{args.out_path}/{args.fDirName}_H5"
os.makedirs(args.out_path, exist_ok=True)

logging.basicConfig(filename=f"{args.out_path}/log.txt", level=logging.DEBUG)

with open("preprocess/createH5s/meta.yaml", 'r') as f:
    meta = yaml.full_load(f)[args.fDirName.split("_")[0]] #only fetch the meta corresponding to the current dataset
    if 'multi_primary' in meta and meta['multi_primary'] and "primary_data_tags" not in meta['desctags']:
        logging.error("Error: multi_primary is set to True but primary_data_tags is not defined in meta.yaml")
        sys.exit(1)

with h5py.File(f"{args.out_path}/data.h5", "w") as h5_file: # create the HDF5 file
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
                    dgroup = h5_file.create_group(f"{patientID}/{fieldID}/{instanceID}")

                    # read manifest.cvs file from inside the zip
                    df = pd.read_csv(glob(f"{tmp_dir}/manifest.*")[0], on_bad_lines='skip') # sometimes the manifest file is manifest.csv and sometimes manifest.cvs
                    n_series = len(df.seriesid.unique())

                    dgroup.attrs["patientDICOMID"] = readSingleValMeta(df, "patientid") if "patientid" in df.columns else ""
                    dgroup.attrs["studyDICOMID"] = readSingleValMeta(df, "studyid") if "studyid" in df.columns else ""
                    dgroup.attrs["studyDesc"] = readSingleValMeta(df, "study discription") if "study discription" in df.columns else ""
                    dgroup.attrs["aet"] = readSingleValMeta(df, "aet") if "aet" in df.columns else ""
                    dgroup.attrs["host"] = readSingleValMeta(df, "host") if "host" in df.columns else ""
                    dgroup.attrs["date"] = readSingleValMeta(df, "date") if "date" in df.columns else ""
                    dgroup.attrs["n_series"] = n_series

                    seriesIDs = sorted(df.seriesid.unique())
                    cmplx_dataM = []
                    cmplx_dataF = []
                    complx_flags = {}
                    desc_tags = []
                    series_dataset = collections.defaultdict(dict)
                    for i, seriesID in enumerate(seriesIDs):
                        # if len(seriesIDs) <= 6: #TODO: remove this debug
                        #     continue
                        series, seriesMeta = ReadSeries(tmp_dir, return_meta=True, taginits2ignore=["0029"], series_ids=seriesID, series2array=False)
                        if len(series) == 0:                            
                            logging.error(f"Dirty DICOM: In {zip_file}, for seriesID: {seriesID} (the {i+1}th series out of {n_series} series), has an issue with the DICOM files. It will be skipped.")
                            continue
                        seriesDesc = df[df.seriesid==seriesID]['series discription'].unique()
                        if len(seriesDesc) > 1:
                            logging.warning(f"Warning: More than one series discription found in {zip_file}, for seriesID: {seriesID}")
                        seriesDesc = seriesDesc[0]                        

                        n_dims = 2 + sum([meta['multi_channel'], meta['is_dynamic'], meta['is_3D']]) #2 spatial dims + dims as per the data
                        assert n_dims == len(series[0].shape), f"Error: While processsing {fileID}, the number of dimensions in the data ({len(series[0].shape)}) does not match the number of dimensions determined from the meta.yaml ({n_dims})"
                        if not meta['is_3D']:
                            series[0] = np.expand_dims(series[0], -3)
                        if not meta['is_dynamic']:
                            series[0] = np.expand_dims(series[0], -4)
                        if not meta['multi_channel']:
                            series[0] = np.expand_dims(series[0], -5)

                        if seriesDesc in meta['desctags']['primary_data']:
                            dsName = "primary" #if len(meta['desctags']['primary_data']) == 1 else "primary_" + meta['desctags']['primary_data_tags'][meta['desctags']['primary_data'].index(seriesDesc)]
                        elif any(seriesDesc in sublist for sublist in meta['desctags']['auxiliary_data']):
                            dsName = "auxiliary_" + meta['desctags']['auxiliary_data_tags'][next((i for i, sublist in enumerate(meta['desctags']['auxiliary_data']) if seriesDesc in sublist), None)]
                        else:
                            logging.error(f"Error: While processsing {fileID}, {seriesDesc} was not found in meta.yaml")
                            continue

                        if 'multi_primary' in meta and meta['multi_primary'] and "primary" in dsName:
                            dsName += "_" + meta['desctags']['primary_data_tags'][next((i for i, sublist in enumerate(meta['desctags']['primary_data']) if seriesDesc in sublist), None)]

                        plane = determine_orientation(seriesMeta[0]['0020|0037'])
                        if 'default_plane' not in meta or plane != meta['default_plane']:
                            dsName += f"_{plane}"

                        if meta['repeat_acq']:
                            dsName += "_0" #It assumes it to be the first one, then if that is taken, it will be _1, _2, etc - which will be calculated!

                        if meta["is_complex"] and "primary" in dsName: #Currently complex is only supported for primary data
                                    
                            if "ORIGINAL\\PRIMARY\\M" in seriesMeta[0]['0008|0008']:  
                                cmplx_dataM.append({
                                    "M": series[0],
                                    "M_seriesID": seriesID,
                                    "M_seriesMeta": seriesMeta[0]
                                })
                                complx_flags["M"] = True
                            elif "ORIGINAL\\PRIMARY\\P" in seriesMeta[0]['0008|0008']:                              
                                cmplx_dataF.append({
                                    "P": series[0],
                                    "P_seriesID": seriesID,
                                    "P_seriesMeta": seriesMeta[0]
                                })
                                complx_flags["P"] = True
                            else:
                                logging.error(f"Error: While processsing {fileID}, DICOM header 0008|0008 did not return ORIGINAL\\PRIMARY\\M or ORIGINAL\\PRIMARY\\P. Check the issue!")
                                continue

                            if "M" in complx_flags and "P" in complx_flags:  
                                cmplx_datum = {**cmplx_dataM.pop(0), **cmplx_dataF.pop(0)}
                                assert cmplx_datum['M_seriesMeta']['0020|0012'] == cmplx_datum['P_seriesMeta']['0020|0012'], "Error: While processsing {fileID}, the acquisition number of the magnitude and phase data are not the same. Check the issue!"
                                cmplx_datum["P"] = np.interp(cmplx_datum["P"], (cmplx_datum["P"].min(), cmplx_datum["P"].max()), (-np.pi, +np.pi))
                                data = cmplx_datum["M"] * np.exp(1j * cmplx_datum["P"])
                                seriesID = {"mag_0": cmplx_datum["M_seriesID"], "phase_0": cmplx_datum["P_seriesID"]}
                                seriesMeta[0] = {"mag_0": cmplx_datum["M_seriesMeta"], "phase_0": cmplx_datum["P_seriesMeta"]}
                                complx_flags = {}
                            else:
                                data = None
                        else:
                            data = series[0]
                            seriesID = {"mag_0": seriesID} #by default, if it's not complex data, it's magnitude only
                            seriesMeta[0] = {"mag_0": seriesMeta[0]}

                        if data is not None:                            
                            if meta["repeat_acq"] and dsName in series_dataset.keys() and seriesDesc in desc_tags:
                                dsName = dsName.replace("_0", f'_{sorted([int(k.split("_")[-1]) for k in series_dataset.keys() if dsName.replace("_0", "_") in k])[-1]+1}')
                            if dsName in series_dataset.keys():
                                if "stack_dim" in meta and data.shape[meta["stack_dim"]] == 1:
                                    series_dataset[dsName]['data'] = np.concatenate((series_dataset[dsName]['data'], data), axis=meta["stack_dim"])
                                    id_in_series = series_dataset[dsName]['data'].shape[meta["stack_dim"]] - 1
                                else:
                                    logging.error(f"Error: While processsing {fileID}, Concatenation error! Currently, one concatenation per series is supported and stack_dim must be supplied.")
                                    continue
                                new_keys = {"mag_0": f"mag_{id_in_series}", "phase_0": f"phase_{id_in_series}"}
                                series_dataset[dsName]["seriesID"].update({new_keys.get(k, k): v for k, v in seriesID.items()})
                                series_dataset[dsName]["DICOMHeader"].update({new_keys.get(k, k): v for k, v in seriesMeta[0].items()})
                            else:
                                series_dataset[dsName]['data'] = data
                                series_dataset[dsName]["seriesID"] = seriesID
                                series_dataset[dsName]["DICOMHeader"] = seriesMeta[0]
                            series_dataset[dsName]["seriesDesc"] = seriesDesc
                            desc_tags.append(seriesDesc)

                    for dsName in series_dataset.keys():
                        dset = dgroup.create_dataset(dsName, data=series_dataset[dsName]['data'])
                        dset.attrs["min_val"] = abs(series_dataset[dsName]['data']).min() if np.iscomplexobj(series_dataset[dsName]['data']) else series_dataset[dsName]['data'].min()
                        dset.attrs["max_val"] = abs(series_dataset[dsName]['data']).max() if np.iscomplexobj(series_dataset[dsName]['data']) else series_dataset[dsName]['data'].max()
                        dset.attrs["seriesID"] = json.dumps(series_dataset[dsName]["seriesID"])
                        dset.attrs["DICOMHeader"] = json.dumps(series_dataset[dsName]["DICOMHeader"])
                        dset.attrs["seriesDesc"] = series_dataset[dsName]["seriesDesc"]

                        
        except Exception as ex:
            logging.error(f"Error: {ex} in {zip_file} at line {sys.exc_info()[-1].tb_lineno}")
            with contextlib.suppress(Exception): #if the fileds like patientID, fieldID, or instanceID are not even tehre, then the exception can safely ignored. If they are thre, the group might already be created - needs to be deleted
                if f"{patientID}/{fieldID}/{instanceID}" in h5_file:
                    del h5_file[f"{patientID}/{fieldID}/{instanceID}"]

    print(f"All Zips processed!")

print("HDF5 closed! Over and out!")