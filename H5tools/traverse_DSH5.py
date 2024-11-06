"""
This script creates the HDF5 files from the zip files downloaded from UK Biobank.
This is a generic version, was used for the creation of the HDF5 files for the short and long axis heart images.
"""
import numpy as np
import json
import h5py
import argparse
import pandas as pd
import os
import json
from collections import defaultdict
import dateutil.parser as dp

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", default=r"", help="path where the HDF5 file is storred")
parser.add_argument("--dataH5", default="data.h5", help="name of the HDF5 file")
parser.add_argument("--mode", type=int, default=6, help="0: fetch all the subject IDs, 1: create subset of the dataset and return the subject IDs based on the number of cardiac cycles (for heart 208 and 209), 2: Get Mean and STD of the dataset, 3: get unique dataset keys and counts, 4: get original filenames before HDF5 conversion, 5: return the date of the MRI acquisition, 6: return the subject IDs based on the address of the medical centre, 7: get the unique shapes")
parser.add_argument("--orig_DICOM", action=argparse.BooleanOptionalAction, default=False, help="whether or not the original dataset used for the HDF5 conversion was in DICOM format")
parser.add_argument("--save_findings", action=argparse.BooleanOptionalAction, default=True, help="path to store the HDF5 file")
args = parser.parse_args()

if args.save_findings:
    os.makedirs(f"{args.in_path}/meta", exist_ok=True)

print(args.dataH5)

with h5py.File(f"{args.in_path}/{args.dataH5}", 'r', swmr=True) as f:    
    if args.mode == 0: ## fetch the subject IDs
        subIDs = list(f.keys())
        if args.save_findings:
            with open(f"{args.in_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}subIDs.json", "w") as f:
                json.dump(subIDs, f)
        else:
            print(f.keys())
            
    elif args.mode == 1: ## create subset of the dataset and return the subject IDs based on the number of cardiac cycles
        subIDs = defaultdict(lambda: defaultdict(list))
        def get_subs(name, obj):
            if isinstance(obj, h5py.Dataset):
                n_cycle = int(eval(obj.attrs['DICOMHeader'])['mag_0']['0020|4000'].split(";")[1].strip().split(" ")[0])
                name_parts = name.split("/")
                subIDs[name_parts[-1]][n_cycle].append(name_parts[0])
        f.visititems(get_subs)
        for tag in subIDs:
            for n_cycle in subIDs[tag]:
                subIDs[tag][n_cycle] = list(set(subIDs[tag][n_cycle]))
        if args.save_findings:
            with open(f"{args.in_path}/meta/subIDs_Acqs_nCardiacCycles.json", "w") as f:
                json.dump(subIDs, f)
        else:
            print(subIDs)
            
    elif args.mode == 2: ## Get Mean and STD of the dataset
        sum = 0.0
        sum_squares = 0.0
        total_elements = 0
        def accumulate_stats(name, obj):
            if isinstance(obj, h5py.Dataset):
                global sum, sum_squares, total_elements
                sum += np.sum(obj)
                sum_squares += np.sum(np.square(obj))
                total_elements += obj.size
        f.visititems(accumulate_stats)
        mean = sum / total_elements
        variance = (sum_squares / total_elements) - (mean**2)  # Use the formula: Var[X] = E[X^2] - (E[X])^2
        std_dev = np.sqrt(variance)
        stats = {"mean": mean, "std_dev": std_dev}
        if args.save_findings:
            with open(f"{args.in_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}stats.json", "w") as f:
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
            with open(f"{args.in_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}unique_keys.json", "w") as f:
                json.dump(counter, f)
                
    elif args.mode == 4: ## fetch the original filenames before HDF5 conversion
        orig_filenames = []
        def get_orig_filenames(name, obj):
            if isinstance(obj, h5py.Dataset):
                orig_filenames.append("_".join(name.split("/")[:-1]))
        f.visititems(get_orig_filenames)
        if args.save_findings:
            with open(f"{args.in_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}orig_filenames.json", "w") as f:
                json.dump(orig_filenames, f)
        else:
            print(f.keys())

    elif args.mode == 5: #return the date of the MRI acquisition
        dates = []
        def get_dates(name, obj):
            if 'date' in obj.attrs and bool(obj.attrs['date']):
                date = dp.parse(str(obj.attrs['date']))
                name_split = name.split(sep='/')
                dates.append((name_split[0],name_split[-1] ,date))
        f.visititems(get_dates)
        dates = pd.DataFrame(dates, columns=["eid", "instance", "date"])
        if args.save_findings:
            dates.to_csv(f"{args.in_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}subIDs_MRIdates.csv", index=False)
        else:
            print(dates)

    elif args.mode == 6: # return the subject IDs based on the address of the medical centre
        subIDs = defaultdict(lambda: defaultdict(list))
        def get_subs(name, obj):
            if args.orig_DICOM and isinstance(obj, h5py.Dataset):
                MRIcentre_name = eval(obj.attrs['DICOMHeader'])['mag_0']['0008|0081']
                name_parts = name.split("/")
                subIDs[name_parts[-1]][MRIcentre_name].append(name_parts[0])
            elif 'host' in obj.attrs and bool(obj.attrs['host']):
                MRIcentre_name = obj.attrs['host']
                name_parts = name.split("/")
                subIDs["primary"][MRIcentre_name].append(name_parts[0])
        f.visititems(get_subs)
        for tag in subIDs:
            for centre in subIDs[tag]:
                subIDs[tag][centre] = list(set(subIDs[tag][centre]))
        if args.save_findings:
            with open(f"{args.in_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}subIDs_Acqs_MRICentre.json", "w") as f:
                json.dump(subIDs, f)
        else:
            print(subIDs)

    elif args.mode == 7: # get the unique shapes
        counter = defaultdict(lambda: defaultdict(int))
        def count_items(name, obj):
            if isinstance(obj, h5py.Dataset):
                counter[name.split("/")[-1]][obj.shape] += 1
        f.visititems(count_items)
        if args.save_findings:
            df = pd.DataFrame(counter)
            df.to_csv(f"{args.in_path}/meta/{'' if args.dataH5=='data.h5' else args.dataH5.replace('.h5','_')}unique_shapes.csv")
        else:
            print(counter)
            
    else:
        counter = []
        def count_items(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.shape[1] != 1:
                counter.append(obj.shape[1])
        f.visititems(count_items)

        print(len(counter))

print("Done!")