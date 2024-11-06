import h5py
import argparse
import pandas as pd
import os
import numpy as np
import time

# read the first command line argument and set the detault to "C:\Users\Public\Documents\HDF5\*.zip" using argparse

def saver(df, args, suffix=""):      
    if args.prep_Zs:
        df.data = df.data.apply(np.squeeze)
        Zs_df = df.data.apply(pd.Series)
        Zs_df.columns = [f'Z{i}' for i in range(len(Zs_df.columns))]
        df = pd.concat([df.drop('data', axis=1), Zs_df], axis=1)

    if args.save_csv:
        df.to_csv(f"{args.out_path}/{args.save_tag}{os.path.basename(args.in_path).replace('.h5','')}{suffix}.csv", index=False)
    else:
        np.save(f"{args.out_path}/{args.save_tag}{os.path.basename(args.in_path).replace('.h5','')}{suffix}.npy", df)

def process_embs(args, save_complex=False, save_npy=True):
    start_time = time.time()

    data = []
    def process_embs(name, obj):
        if isinstance(obj, h5py.Dataset):
            name_parts = name.split("/")
            if "subIDs" in args and bool(args.subIDs) and name_parts[0] not in args.subIDs:
                return
            if "data_tags" in args and bool(args.data_tags) and name_parts[-1] not in args.data_tags:
                return
            datum = {
                "subID": name_parts[0],
                "fieldID": name_parts[1],
                "instanceID": name_parts[2],
                "data_tag": name_parts[-1],
                "data": obj[()].flatten() if "flatten_data" in args and args.flatten_data else obj[()]
            }
            data.append(datum)

    if ".h5" not in args.in_path:
        args.in_path = f"{args.in_path}/emb.h5"
        
    with h5py.File(args.in_path, 'r', swmr=True) as f:   
        f.visititems(process_embs)

    end_time = time.time()
    print("Time taken (for fetching): ", time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))

    df = pd.DataFrame(data)
    print("emb.h5 processing Done!")

    if save_npy:
        start_time = time.time()
        if not bool(args.out_path):
            args.out_path = os.path.dirname(args.in_path)

        if (not np.iscomplexobj(df['data'][0])) or save_complex:        
            saver(df, args)
        else:
            print("Complex data detected! Saving subparts (Real, Imaginary, Magnitude, and Phase) separately...")
            subparts = {"real": np.real, "imag": np.imag, "mag": np.abs, "phase": np.angle}        
            for k, v in subparts.items():
                df_tmp = df.copy()
                df_tmp["data"] = df_tmp["data"].apply(v)
                saver(df_tmp, args, suffix=f"_{k}")
            
        end_time = time.time()
        print("Time taken (for saving): ", time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))
    else:
        if args.prep_Zs:
            df.data = df.data.apply(np.squeeze)
            Zs_df = df.data.apply(pd.Series)
            Zs_df.columns = [f'Z{i}' for i in range(len(Zs_df.columns))]
            df = pd.concat([df.drop('data', axis=1), Zs_df], axis=1)
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", help="Path to the HDF5 file of the embeddings. Can be a direct .h5 file, or folder containing emb.h5", default=r"")
    parser.add_argument("--out_path", help="Path to the store the embedding CSV. Can leave it blank to store inside the same directory as the HDF5", default=r"")
    parser.add_argument("--save_tag", help="Tag to put at the front of the output file name.", default=r"")

    #filers
    parser.add_argument("--subIDs", help="Coma-seperated list of subject IDs. Blank for all.", default=r"")
    parser.add_argument("--data_tags", help="Coma-seperated list of data tags. Blank for all.", default=r"primary_LAX_4Ch_transverse_0")
    parser.add_argument("--flatten_data", action=argparse.BooleanOptionalAction, help="Whether to flatten the data before storing or not.", default=False)
    parser.add_argument("--save_csv", action=argparse.BooleanOptionalAction, help="Whether to save as CSV or as an NPY file.", default=False)
    parser.add_argument("--prep_Zs", action=argparse.BooleanOptionalAction, help="Whether to prep the Zs by splitting into different cols.", default=False)

    args = parser.parse_args()
    process_embs(args)


