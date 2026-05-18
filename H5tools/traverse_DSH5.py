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
import dateutil.parser as dp
import nibabel as nib
from PIL import Image

# read the first command line argument and set the detault to "C:\Users\Public\Documents\HDF5\*.zip" using argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", default=r"/group/glastonbury/soumick/toysets/dummy_F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3", help="path where the HDF5 file is storred")
parser.add_argument("--dataH5", default="data.h5", help="name of the HDF5 file")
# parser.add_argument("--in_path", default=r"/scratch/glastonbury/datasets/ukbbH5s/F20259_Pancreas_Images_ShMoLLI_DICOM_H5v3", help="path where the HDF5 file is storred")
# parser.add_argument("--dataH5", default="data.h5", help="name of the HDF5 file")
parser.add_argument("--mode", type=int, default=0, help="0: fetch all the subject IDs, " \
                                                        "1: create subset of the dataset and return the subject IDs based on the number of cardiac cycles (for heart 208 and 209), " \
                                                        "2: Get Mean and STD of the dataset, " \
                                                        "3: get unique dataset keys and counts, " \
                                                        "4: get original filenames before HDF5 conversion, " \
                                                        "5: return the date of the MRI acquisition, " \
                                                        "6: return the subject IDs based on the address of the medical centre, " \
                                                        "7: get the unique shapes, " \
                                                        "8: export the images as individual files (export format is determined by the --export_format argument, default is NIFTI)")
parser.add_argument("--orig_DICOM", action=argparse.BooleanOptionalAction, default=True, help="whether or not the original dataset used for the HDF5 conversion was in DICOM format")
parser.add_argument("--save_findings", action=argparse.BooleanOptionalAction, default=True, help="path to store the HDF5 file")
parser.add_argument("--export_format", type=str, default="NIFTI", help="export format for the images, default is NIFTI. Other options are 'PNG', 'JPEG', 'TIFF', 'BMP', 'GIF'")
parser.add_argument("--export_N", type=int, default=0, help="export only N images, default is 0. This is only used when the mode is 8. If 0 or -1 or None, all images are exported")
parser.add_argument("--export_key", type=str, default="", help="export key for the images, default is blank = use all.")

parser.add_argument("--out_path", default="", help="path where the outputs will be stored (subfolder with the final folder name of the in_path will be created). If blank, the in_path will be used.")

#filters
parser.add_argument('--ds_names_present', type=str, action="store", default="", help="if it is not empty (coma-separated or OR-seperated list), then only patients with this dataset name will be used [except for mode 3]")

args = parser.parse_args()

if bool(args.out_path):
    args.out_path = os.path.join(args.out_path, os.path.basename(args.in_path))
else:
    args.out_path = args.in_path

if args.save_findings:
    if args.mode == 8:
        os.makedirs(f"{args.out_path}/exported_images/{args.export_format}", exist_ok=True)
    else:
        os.makedirs(f"{args.out_path}/meta", exist_ok=True)

print(args.dataH5)

with h5py.File(f"{args.in_path}/{args.dataH5}", 'r', swmr=True) as f:   
         
    sub_with_DS = []      
    if args.ds_names_present != "":  
        print("Filtering to check if the supplied datasets are mentioned...")
        ds_names_present_list = args.ds_names_present.split(',') if ',' in args.ds_names_present else args.ds_names_present.split('OR')
        def dsFind(name, obj):
            if isinstance(obj, h5py.Dataset):
                for dsname in ds_names_present_list:
                    if dsname in name:
                        sub_with_DS.append(name.split('/')[0])
                        break
        f.visititems(dsFind)
    if args.ds_names_present != "":
        if len(ds_names_present_list) == 1:
            tag =  f'_{ds_names_present_list[0]}'
        else:
            tag = f'_{"OR".join(ds_names_present_list)}'
    else:
        tag = ""
    sub_with_DS = set(sub_with_DS)

    if args.mode == 0: ## fetch the subject IDs
        subIDs = list(sub_with_DS) if args.ds_names_present != "" else list(f.keys())
        if args.save_findings:
            with open(f"{args.out_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}subIDs{tag}.json", "w") as f:
                json.dump(subIDs, f)
        else:
            print(f.keys())
    elif args.mode == 1: ## create subset of the dataset and return the subject IDs based on the number of cardiac cycles
        subIDs = defaultdict(lambda: defaultdict(list))
        def get_subs(name, obj):
            if isinstance(obj, h5py.Dataset):
                n_cycle = int(eval(obj.attrs['DICOMHeader'])['mag_0']['0020|4000'].split(";")[1].strip().split(" ")[0])
                name_parts = name.split("/")
                if sub_with_DS and name_parts[0] not in sub_with_DS:
                    return
                subIDs[name_parts[-1]][n_cycle].append(name_parts[0])
        f.visititems(get_subs)
        for stag in list(subIDs.keys()):           
            for n_cycle in list(subIDs[stag].keys()):  
                subIDs[stag][n_cycle] = list(set(subIDs[stag][n_cycle]))
        if args.save_findings:
            with open(f"{args.out_path}/meta/subIDs_Acqs_nCardiacCycles{tag}.json", "w") as f:
                json.dump(subIDs, f)
        else:
            print(subIDs)
    elif args.mode == 2: ## Get Mean and STD of the dataset
        sum_total = 0.0
        sum_squares = 0.0
        total_elements = 0
        def accumulate_stats(name, obj):
            if isinstance(obj, h5py.Dataset):
                name_parts = name.split("/")
                if sub_with_DS and name_parts[0] not in sub_with_DS:
                    return
                global sum_total, sum_squares, total_elements
                data = obj[()]
                sum_total += np.sum(data)
                if np.issubdtype(data.dtype, np.complexfloating): # Sum of squared magnitudes for complex data
                    sum_squares += np.sum(np.abs(data)**2)
                else:                    
                    sum_squares += np.sum(data**2) # Regular sum of squares for real data
                total_elements += obj.size
        f.visititems(accumulate_stats)
        mean = sum_total / total_elements
        variance = (sum_squares / total_elements) - np.abs(mean)**2  # Use the formula: Var[X] = E[X^2] - (E[X])^2
        std_dev = np.sqrt(variance.real)
        if np.iscomplexobj(mean):
            stats = {"mean_real": mean.real, "mean_imag": mean.imag, "mean_mag": np.abs(mean), "mean_phase": np.angle(mean), "std_dev": std_dev}
        else:
            stats = {"mean": mean, "std_dev": std_dev}
        if args.save_findings:
            with open(f"{args.out_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}stats{tag}.json", "w") as f:
                json.dump(stats, f)
        else:
            print(stats)        
    elif args.mode == 3: #get unique dataset keys and counts
        counter = defaultdict(int)
        def count_items(name, obj):
            if isinstance(obj, h5py.Dataset):
                counter[name.split("/")[-1]] += 1
        f.visititems(count_items)
        if args.save_findings:
            with open(f"{args.out_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}unique_keys.json", "w") as f:
                json.dump(counter, f)
                
    elif args.mode == 4: ## fetch the original filenames before HDF5 conversion
        orig_filenames = []
        def get_orig_filenames(name, obj):
            if isinstance(obj, h5py.Dataset):
                name_parts = name.split("/")
                if sub_with_DS and name_parts[0] not in sub_with_DS:
                    return
                orig_filenames.append("_".join(name.split("/")[:-1]))
        f.visititems(get_orig_filenames)
        if args.save_findings:
            with open(f"{args.out_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}orig_filenames{tag}.json", "w") as f:
                json.dump(orig_filenames, f)
        else:
            print(f.keys())

    elif args.mode == 5: #return the date of the MRI acquisition
        dates = []
        def get_dates(name, obj):
            name_parts = name.split("/")
            if sub_with_DS and name_parts[0] not in sub_with_DS:
                return
            if 'date' in obj.attrs and bool(obj.attrs['date']):
                date = dp.parse(str(obj.attrs['date']))
                name_split = name.split(sep='/')
                dates.append((name_split[0],name_split[-1] ,date))
        f.visititems(get_dates)
        dates = pd.DataFrame(dates, columns=["eid", "instance", "date"])
        if args.save_findings:
            dates.to_csv(f"{args.out_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}subIDs_MRIdates{tag}.csv", index=False)
        else:
            print(dates)

    elif args.mode == 6: # return the subject IDs based on the address of the medical centre
        subIDs = defaultdict(lambda: defaultdict(list))
        def get_subs(name, obj):
            name_parts = name.split("/")
            if sub_with_DS and name_parts[0] not in sub_with_DS:
                return
            if args.orig_DICOM and isinstance(obj, h5py.Dataset):
                MRIcentre_name = eval(obj.attrs['DICOMHeader'])['mag_0']['0008|0081']
                subIDs[name_parts[-1]][MRIcentre_name].append(name_parts[0])
            elif 'host' in obj.attrs and bool(obj.attrs['host']):
                MRIcentre_name = obj.attrs['host']
                subIDs["primary"][MRIcentre_name].append(name_parts[0])
        f.visititems(get_subs)
        for stag in list(subIDs.keys()):           
            for centre in list(subIDs[stag].keys()):  
                subIDs[stag][centre] = list(set(subIDs[stag][centre]))
        if args.save_findings:
            with open(f"{args.out_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}subIDs_Acqs_MRICentre{tag}.json", "w") as f:
                json.dump(subIDs, f)
        else:
            print(subIDs)

    elif args.mode == 7: # get the unique shapes
        counter = defaultdict(lambda: defaultdict(int))
        def count_items(name, obj):
            if isinstance(obj, h5py.Dataset):
                name_parts = name.split("/")
                if sub_with_DS and name_parts[0] not in sub_with_DS:
                    return
                counter[name.split("/")[-1]][obj.shape] += 1
        f.visititems(count_items)
        if args.save_findings:
            df = pd.DataFrame(counter)
            df.to_csv(f"{args.out_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}unique_shapes{tag}.csv")
        else:
            print(counter)

    elif args.mode == 8: # export the images as individual files
        export_format = args.export_format.upper()
        
        counter = 0
        if bool(args.export_N) and args.export_N != -1 and args.export_N is not None:
            print(f"Exporting only {args.export_N} images in {export_format} format.")

        def export_image(name, obj):
            global counter
            if bool(args.export_N) and args.export_N != -1 and args.export_N is not None and counter >= args.export_N:
                return
            if isinstance(obj, h5py.Dataset):
                name_parts = name.split("/")
                if sub_with_DS and name_parts[0] not in sub_with_DS:
                    return
                if bool(args.export_key) and args.export_key not in name:
                    return
                data = obj[()]
                if np.iscomplexobj(data):
                    data = np.abs(data) # Convert complex data to magnitude, future TODO phase seperately
                data = np.transpose(data, (3, 4, 2, 0, 1)) #from [channel, time, slice, height, width] to [height, width, slice, channel, time]
                ndim = data.squeeze().ndim
                if export_format in ["NIFTI", "NII"]:
                    data = np.transpose(data, (1, 0, 2, 3, 4))
                    img = nib.Nifti1Image(data.squeeze(), np.eye(4))
                    nib.save(img, f"{args.out_path}/exported_images/{args.export_format}/{name.replace('/', '_')}.nii.gz")
                elif export_format in ["TIFF", "TIF", "GIF"]:
                    # PIL supports up to 3D (multi-page for 3D)
                    if ndim == 2:
                        img = Image.fromarray(data.squeeze())
                        ext = "tiff" if export_format in ["TIFF", "TIF"] else "gif"
                        img.save(f"{args.out_path}/exported_images/{args.export_format}/{name.replace('/', '_')}.{ext}")
                    elif ndim == 3:
                        # Save as multi-page TIFF or GIF
                        imgs = [Image.fromarray(data.squeeze()[...,i]) for i in range(data.shape[-1])]
                        ext = "tiff" if export_format in ["TIFF", "TIF"] else "gif"
                        imgs[0].save(
                            f"{args.out_path}/exported_images/{args.export_format}/{name.replace('/', '_')}.{ext}",
                            save_all=True, append_images=imgs[1:]
                        )
                    else:
                        # For 4D/5D, iterate and save slices as separate files
                        idx = np.ndindex(data.shape[:-2])
                        ext = "tiff" if export_format in ["TIFF", "TIF"] else "gif"
                        for index in idx:
                            slice2d = data[index]
                            img = Image.fromarray(slice2d)
                            idx_str = "_".join([f"d{i}_{v}" for i, v in enumerate(index)])
                            img.save(f"{args.out_path}/exported_images/{args.export_format}/{name.replace('/', '_')}_{idx_str}.{ext}")
                else:
                    # For PNG, JPEG, BMP, etc.
                    # Save each 2D slice as a separate file
                    if ndim == 2:
                        slice_data = data
                        if np.issubdtype(slice_data.dtype, np.floating):
                            slice_min = np.nanmin(slice_data)
                            slice_max = np.nanmax(slice_data)
                            scaled = (255 * (slice_data - slice_min) / (slice_max - slice_min + 1e-8)).astype(np.uint8)
                        else:
                            scaled = slice_data
                        img = Image.fromarray(scaled)
                        img.save(f"{args.out_path}/exported_images/{args.export_format}/{name.replace('/', '_')}.{export_format.lower()}")
                    else:
                        idx = np.ndindex(data.shape[:-2])
                        for index in idx:
                            slice2d = data[index]
                            if np.issubdtype(slice2d.dtype, np.floating):
                                slice_min = np.nanmin(slice2d)
                                slice_max = np.nanmax(slice2d)
                                scaled = (255 * (slice2d - slice_min) / (slice_max - slice_min + 1e-8)).astype(np.uint8)
                            else:
                                scaled = slice2d
                            idx_str = "_".join([f"d{i}_{v}" for i, v in enumerate(index)])
                            img = Image.fromarray(scaled)
                            img.save(f"{args.out_path}/exported_images/{args.export_format}/{name.replace('/', '_')}_{idx_str}.{export_format.lower()}")
                counter += 1
        f.visititems(export_image)
    else:
        counter = []
        def count_items(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.shape[1] != 1:
                name_parts = name.split("/")
                if sub_with_DS and name_parts[0] not in sub_with_DS:
                    return
                counter.append(obj.shape[1])
        f.visititems(count_items)

        print(len(counter))

print("Done!")