import os
from os.path import join as pjoin
from glob import glob
import pandas as pd

import numpy as np
import h5py
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def H5_to_DF(h5_path, valcolnames):
    data_list = []

    def collect_data(name, obj):
        if isinstance(obj, h5py.Dataset):
            SubID, FieldID, InstanceID, Acq = name.split('/')
            data = np.squeeze(obj[()])
            data_list.append([SubID, FieldID, InstanceID, Acq] + list(data))

    with h5py.File(h5_path, 'r') as file:
        file.visititems(collect_data)

    df = pd.DataFrame(data_list, columns=['SubID', 'FieldID', 'InstanceID', 'Acq'] + valcolnames)
    df.set_index('SubID', inplace=True)
    df.index = df.index.astype(int)
    return df


def process_item(metrics_path, args, train_set, val_set, pheno_df=None):    
    if args.create_vis:
        print(f"Creating folder to store visualizations for {metrics_path}")
        os.makedirs(metrics_path.replace("metrics.csv", "vis"), exist_ok=True)

    #Process the metrics file if it does not exist
    if args.force_process_metrics or not os.path.exists(metrics_path.replace("metrics.csv", "metrics_wSplits.csv")):
        print(f"Processing {metrics_path}")
        try:
            metrics_df = pd.read_csv(metrics_path)
            metrics_df['SubID'] = metrics_df['SubID'].astype(str)

            metrics_df['Split'] = 'Test'
            metrics_df.loc[metrics_df['SubID'].isin(train_set), 'Split'] = 'Train'
            metrics_df.loc[metrics_df['SubID'].isin(val_set), 'Split'] = 'Validation'

            metrics_df.to_csv(metrics_path.replace("metrics.csv", "metrics_wSplits.csv"), index=False)
            
            metrics_df.set_index('SubID', inplace=True)
            metrics_df.drop(columns=set(metrics_df.columns).intersection(['Channel', 'Slice', 'FieldID', "InstanceID"]), axis=1, inplace=True)
            metrics_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            metrics_df.dropna(inplace=True)

            valid_metrics_df = metrics_df.select_dtypes(include=['float', 'int']).loc[:, ~(metrics_df == -1).all()].join(metrics_df['Split'])
            median = valid_metrics_df.groupby(['Split']).median().round(4).astype(str) +"±"+ valid_metrics_df.groupby(['Split']).mad().round(4).astype(str)
            median.index = "Median_" + median.index.astype(str)
            mean = valid_metrics_df.groupby(['Split']).mean().round(4).astype(str) +"±"+ valid_metrics_df.groupby(['Split']).std().round(4).astype(str)
            mean.index = "Mean_" + mean.index.astype(str)
            consolidated_metrics_df = pd.concat([mean, median])
            consolidated_metrics_df.to_csv(metrics_path.replace("metrics.csv", "metrics_wSplits_consolidated.csv"))

            if args.create_vis:
                print(f"Creating visualizations for {metrics_path}")
                for metric in consolidated_metrics_df.columns:
                    sns.violinplot(data=valid_metrics_df, x="Split", y=metric)
                    plt.tight_layout()
                    plt.savefig(metrics_path.replace("metrics.csv", f"vis/{metric}.{args.vis_ext}"))
                    plt.close()

        except Exception as e:
            print(f"Error in: {metrics_path}: {e}")

    #Process the phenotype file if it does not exist
    if bool(args.pheno_path) and os.path.exists(metrics_path.replace("metrics.csv", "pheno_pred.h5")) and (args.force_process_pheno or not os.path.exists(metrics_path.replace("metrics.csv", "phenometrics_wSplits.csv"))):
        print(f"Processing Phenotype Prediction Results of {metrics_path}")
        try:
            df = H5_to_DF(metrics_path.replace("metrics.csv", "pheno_pred.h5"), args.valcols)
            df.to_csv(metrics_path.replace("metrics.csv", "pheno_pred_raw.tsv"), sep='\t')
            scaler = StandardScaler()
            for column in pheno_df.columns:
                _ = scaler.fit(pheno_df[[column]])
                df[column] = scaler.inverse_transform(df[[column]])
            df.to_csv(metrics_path.replace("metrics.csv", "pheno_pred_unscaled.tsv"), sep='\t')
            merged_df = pheno_df.merge(df, left_index=True, right_index=True, suffixes=('_actual', '_predicted'))

            merged_df.index = merged_df.index.astype(str)
            merged_df['Split'] = 'Test'
            merged_df.loc[merged_df.index.isin(train_set), 'Split'] = 'Train'
            merged_df.loc[merged_df.index.isin(val_set), 'Split'] = 'Validation'

            metrics_df = pd.DataFrame(columns=['RMSE', 'R^2', 'r'])
            for split in ['Train', 'Validation', 'Test']:
                for column in pheno_df.columns:
                    actual = merged_df[merged_df['Split']==split][f"{column}_actual"]
                    predicted = merged_df[merged_df['Split']==split][f"{column}_predicted"]

                    r2 = r2_score(actual, predicted)
                    r, _ = pearsonr(actual, predicted)
                    rmse = mean_squared_error(actual, predicted, squared=False)
                    metrics_df.loc[f"{column}_{split}"] = [rmse, r2, r]

                    if args.create_vis:
                        plt.figure(figsize=(8, 8))
                        sns.scatterplot(x=actual, y=predicted)
                        plt.xlabel('Actual Values')
                        plt.ylabel('Predicted Values')
                        plt.title(f'Scatter plot of Actual vs Predicted Values for {column} ({split})')
                        plt.tight_layout()
                        plt.savefig(metrics_path.replace("metrics.csv", f"vis/pheno_pred_scatter_{split}_{column}.{args.vis_ext}"))
                        plt.close()
            metrics_df.to_csv(metrics_path.replace("metrics.csv", "phenometrics_wSplits.csv"))
        except Exception as e:
            print(f"Error in: {metrics_path}: {e}")

    if args.n_recon_GIFs > 0:
        recons = []
        with h5py.File("../Out/Results/F20208_heart_1Ses_time2slc_MskCrop128_V2_3D_ph1_ce1rc1va1_4ChTrans128fold0_precbf16-mixed_pythaemodel-custom_ultra_cevae/Output_fullDS/recon.h5", "r") as f:
            keys = sorted(f.keys())
            for i in range(args.n_recon_GIFs):
                recon = f[keys[i]]
                recons.append(np.array(f[keys[i]]['20208']['2_0']['primary_LAX_4Ch_transverse_0']))

def getARGSParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", help="path to the trainings", default="../Out/Results")
    parser.add_argument("--trainID", help="Leave it blank if all the trainings in the res_path are to be analysed", default="")

    parser.add_argument("--pheno_path", help="path to the file to be used for evaluating phenotypes", default="")
    parser.add_argument("--valcols", help="path to the file to be used for evaluating phenotypes", default="")

    parser.add_argument("--split_csv_path", help="path to the fold CSV", default="../datasets/F20208_Long_axis_heart_images_DICOM_H5/5folds_1Ses_100DS_60Trn_10Val_30Tst_1701seed.csv")
    parser.add_argument("--foldID", help="path to the zip files", default=0, type=int)

    parser.add_argument('--force_process_metrics', action=argparse.BooleanOptionalAction, default=False, help="Process the metric files even if they exist")
    parser.add_argument('--force_process_pheno', action=argparse.BooleanOptionalAction, default=False, help="Process the phenotype files even if they have been processed earlier")
    parser.add_argument('--create_vis', action=argparse.BooleanOptionalAction, default=True, help="Create visualisations for the metrics")
    parser.add_argument('--vis_ext', default="png", help="File extension of the visualisations for saving")

    parser.add_argument("--n_recon_GIFs", help="If greater than one, first n recons will be saved as GIFs", default=0, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = getARGSParser()
    if bool(args.trainID):
        metrics_paths = glob(pjoin(args.res_path, args.trainID, "**", "metrics.csv"), recursive=True)
    else:
        metrics_paths = glob(pjoin(args.res_path, "**", "metrics.csv"), recursive=True)

    split_df = pd.read_csv(args.split_csv_path)
    fold_df = split_df.iloc[args.foldID]     
    train_set = set(eval(fold_df['train']))
    val_set = set(eval(fold_df['val']))

    if bool(args.pheno_path):
        args.valcols = args.valcols.split(',')
        pheno_df = pd.read_table(args.pheno_path, index_col="FID")[args.valcols]
    else:
        pheno_df = None

    for metrics_path in metrics_paths:
        process_item(metrics_path, args, train_set, val_set, pheno_df)
