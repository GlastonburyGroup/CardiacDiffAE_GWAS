import random
import numpy as np
import h5py
import pandas as pd
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

#basic
parser.add_argument('--path_h5s', action="store", default=r"data.h5")

parser.add_argument('--seed', type=int, action="store", default=1701)
parser.add_argument('--per_dataset', type=float, action="store", default=1, help="percent of the dataset to use")
parser.add_argument('--per_train', type=float, action="store", default=0.75)
parser.add_argument('--per_val', type=float, action="store", default=0.10)
parser.add_argument('--per_test', type=float, action="store", default=0.15)
parser.add_argument('--n_folds', type=int, action="store", default=5)

#filters
parser.add_argument('--patient_n_sessions', type=int, action="store", default=1, help="if it is 1, then only patients with 1 session will be used. If 2, then only patients with 2 sessions will be used. If 0 or -1, then all")
parser.add_argument('--ds_names_present', type=str, action="store", default="primary_0", help="if it is not empty (coma-separated list), then only patients with this dataset name will be used")

args, _ = parser.parse_known_args()

random.seed(args.seed)
np.random.seed(args.seed)


patientIDs_master = []
for path_h5 in args.path_h5s.split(','):
    with h5py.File(path_h5, 'r') as f:
        patientIDs = []
        if args.patient_n_sessions > 0:
            # Filter the patients based on the number of sessions
            def count_items(name, obj):
                if name.count('/') == 1 and isinstance(obj, h5py.Group):
                    n_items = len(obj.keys())
                    if n_items == args.patient_n_sessions:
                        patientIDs.append(name.split("/")[0])
            f.visititems(count_items)
        else:
            patientIDs = list(f.keys())

        if args.ds_names_present != "":  
            ds_names_present = args.ds_names_present.split(',')
            _patientIDs = []      
            def dsFind(name, obj):
                if isinstance(obj, h5py.Dataset):
                    for dsname in ds_names_present:
                        if dsname in name:
                            _patientIDs.append(name.split('/')[0])
                            break
            f.visititems(dsFind)
            patientIDs = list(set(patientIDs).intersection(set(_patientIDs)))
        patientIDs_master.append(set(patientIDs))

patientIDs = list(set.intersection(*patientIDs_master))

folds = {}
n_patients = len(patientIDs)

for i in tqdm(range(args.n_folds)):

    # Shuffle the patient IDs
    random.shuffle(patientIDs)

    if bool(args.per_dataset) and args.per_dataset < 1:
        n_patients = int(args.per_dataset * n_patients)
        patientIDs = patientIDs[:n_patients]

    # Split the patient IDs into train, val, and test sets
    n_train = int(args.per_train * n_patients)
    n_val = int(args.per_val * n_patients)
    n_test = n_patients - n_train - n_val

    trainIDs = patientIDs[:n_train] if n_train>0 else []
    valIDs = patientIDs[n_train:n_train+n_val] if n_val>0 else []
    testIDs = patientIDs[n_train+n_val:] if n_test>0 else []

    fold = {
        'train': trainIDs,
        'val': valIDs,
        'test': testIDs
    }

    folds[i] = fold

# Save the folds
for path_h5 in args.path_h5s.split(','):
    out_dir = os.path.dirname(path_h5)
    os.makedirs(out_dir, exist_ok=True)

    out_file = f'{args.n_folds}folds'
    if args.patient_n_sessions > 0:
        out_file += f'_{args.patient_n_sessions}Ses'
    if args.ds_names_present != "":
        if len(ds_names_present) == 1:
            out_file += f'_{ds_names_present[0]}'
        else:
            out_file += f'_{"OR".join(ds_names_present)}'
    out_file += f'_{int(args.per_dataset*100)}DS_{int(args.per_train*100)}Trn_{int(args.per_val*100)}Val_{int(args.per_test*100)}Tst_{args.seed}seed.csv'

    path_folds = os.path.join(out_dir, out_file)
    df = pd.DataFrame.from_dict(folds, orient='index')
    df.to_csv(path_folds, index=False)
