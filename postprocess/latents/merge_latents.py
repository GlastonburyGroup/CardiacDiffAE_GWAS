import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #TODO: find a better way to do this!

from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from sklearn.cross_decomposition import CCA

def align_latents(models):
    aligned_dfs = []
    base_df = models[0]
    
    for df in models[1:]:
        aligned_columns = []
        
        # Calculate the similarity matrix
        similarity_matrix = abs(cosine_similarity(base_df.T, df.T))
        
        # Use the Hungarian Algorithm to find the optimal alignment
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        for base_col, aligned_col in zip(row_ind, col_ind):
            aligned_column = df.iloc[:, aligned_col]
            aligned_column.name = base_df.columns[base_col]
            aligned_columns.append(aligned_column)
        
        aligned_df = pd.concat(aligned_columns, axis=1)
        aligned_dfs.append(aligned_df)
        
    return [base_df] + aligned_dfs

def mcca(models, n_latents):
    n_models = len(models)
    
    for i in range(n_models):
        if type(models[i]) is pd.DataFrame:
            models[i] = models[i].to_numpy()
    
    print("Calculating cross-covariance matrices between all pairs of models...")
    cross_covs = []
    for i in range(n_models):
        for j in range(i, n_models):
            cov = np.dot(models[i].T, models[j]) / (len(models[i]) - 1)
            cross_covs.append(cov)
    
    print("Creating block diagonal matrix....")
    D = np.zeros((n_latents * n_models, n_latents * n_models))
    for i, cov_matrix in enumerate(cross_covs):
        row = i // n_models
        col = i % n_models
        D[row*n_latents:(row+1)*n_latents, col*n_latents:(col+1)*n_latents] = cov_matrix
        if row != col:
            D[col*n_latents:(col+1)*n_latents, row*n_latents:(row+1)*n_latents] = cov_matrix.T

    print("Creating identity matrices block (I ⊗ I) where I is the identity matrix of size (n_latents x n_latents)...")
    I = np.identity(n_latents)
    I_block = np.kron(np.eye(n_models), I)

    print("Solving generalised eigenvalue problem...")
    eigenvalues, eigenvectors = eigh(D, I_block)
    
    print("Sorting eigenvectors based on eigenvalues....")
    sorted_indices = np.argsort(-eigenvalues)  
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = sorted_eigenvectors[:, :n_latents]
    
    print("Extracting canonical components for each model....")
    canonical_components = []
    for i in range(n_models):
        start = i * n_latents
        end = (i + 1) * n_latents
        weights = top_eigenvectors[start:end, :]
        components = np.dot(models[i], weights)
        canonical_components.append(components)

    return canonical_components

def pairwiseCCA(models, n_latents, max_iter=10000):
    n_models = len(models)
    
    for i in range(n_models):
        if type(models[i]) is pd.DataFrame:
            models[i] = models[i].to_numpy()
    
    cca = CCA(n_components=n_latents, max_iter=max_iter)
    
    print("Calculating pairwise CCAs...")
    final_latent = models[0]
    for i in range(1, len(models)):
        print(f"CCA between previous and model {i}")
        cca.fit(final_latent, models[i])
        X_c, Y_c = cca.transform(final_latent, models[i])
        final_latent = (X_c + Y_c) / 2.0

    return final_latent    

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trainIDs', type=str, default="F20208_heart_1Ses_time2slc_MskCrop128_V2_3D100ep_L1_4ChTrans128fold0_prec32_pythaemodel-vae,\
                        F20208_heart_1Ses_time2slc_MskCrop128_V2_3D100ep_seed1993_L1_4ChTrans128fold0_prec32_pythaemodel-vae,\
                        F20208_heart_1Ses_time2slc_MskCrop128_V2_3D100ep_seed42_L1_4ChTrans128fold0_prec32_pythaemodel-vae,\
                        F20208_heart_1Ses_time2slc_MskCrop128_V2_3D100ep_seed1994_L1_4ChTrans128fold0_prec32_pythaemodel-vae,\
                        F20208_heart_1Ses_time2slc_MskCrop128_V2_3D100ep_seed2023_L1_4ChTrans128fold0_prec32_pythaemodel-vae", help='Coma-seperated list of trainIDs to merge')
    
    parser.add_argument("--train_root", type=str, default="../Out/Results", help="Root folder where the trainings are stored")
    parser.add_argument("--res_root", type=str, default="", help="Root folder where the results are stored (Only if train_root is not provided). Will ignore output_folder params. [Not implemented yet]")
    
    parser.add_argument("--gwas_folder", type=str, default="GWAS_fullDSV2", help="Folder inside train_root/trainID where the processed_raw.tsv is available (output of GWAS engager).")
    
    parser.add_argument('--path4res', type=str, default="../merged_latents_models/latents", help='Path where the merged stuff will be saved.')
    parser.add_argument('--save_tag', type=str, default="5Seeds_f0_VAE_128_Crop3D_DSV2", help='Tag to identify this merge')
    
    parser.add_argument('--pre_align', action='store_true', help='Whether to align the latents before merging. [Default: False]')
    parser.add_argument('--merge_method', type=float, default=0, help='0: MCCA, 1: Pairwise CCA, 2: PCA [Default: 0]')
    parser.add_argument('--merge_mcca_with_pca', action='store_true', help='Whether to merge the MCCA components (merge_method 0) with PCA. [Default: False]')
    
    return parser

if __name__ == "__main__":
    args = getARGSParser().parse_args()

    args.trainIDs = [t.strip() for t in args.trainIDs.split(',')]
    
    args.pre_align = True
    if args.pre_align:
        args.path4res = f'{args.path4res}_align'
    
    match args.merge_method:
        case 0:
            args.path4res = f"{args.path4res}_MCCA"
            if args.merge_mcca_with_pca:
                args.path4res = f"{args.path4res}_mergeWithPCA"
            else:
                args.path4res = f"{args.path4res}_mergeWithAvg"
        case 1:
            args.path4res = f"{args.path4res}_pairwiseCCA"
        case 2:
            args.path4res = f"{args.path4res}_PCA"
        case _:
            raise NotImplementedError(f"Merge method ID {args.merge_method} is not implemented yet!")    
    
    args.path4res = f'{args.path4res}/{args.save_tag}' 
    os.makedirs(args.path4res, exist_ok=True)
    
    print("Reading the latents.....")
    processed_latents = []
    for trainID in tqdm(args.trainIDs):
        df = pd.read_table(f'{args.train_root}/{trainID}/{args.gwas_folder}/processed_raw.tsv').sort_values("FID").reset_index(drop=True)
        IID = df.IID
        FID = df.FID
        df.drop(columns=["IID", "FID"], inplace=True)
        latents = df.columns.to_list()
        processed_latents.append(df)

    if args.pre_align:
        print("Aligning the latents....")
        processed_latents = align_latents(processed_latents)
        
    match args.merge_method:
        case 0:
            print("Merging the latents using MCCA....")
            print("Bringing the latents to the canonical components....")
            canonical_components = mcca(processed_latents, n_latents=len(latents))  
            print("Merging the cannonical components....")  
            if args.merge_mcca_with_pca:
                print("....with PCA")
                concatenated_canonical_components = np.concatenate(canonical_components, axis=1)
                pca = PCA(n_components=len(latents))
                final_components = pca.fit_transform(concatenated_canonical_components)
            else:
                print("....with average")
                stacked_canonical_components = np.stack(canonical_components, axis=2)
                final_components = np.mean(stacked_canonical_components, axis=2)
        case 1:
            print("Merging the latents using pairwise CCA....")
            final_components = pairwiseCCA(processed_latents, n_latents=len(latents))
        case 2:
            print("Merging the latents with PCA...")
            concatenated_latents = np.concatenate(processed_latents, axis=1)
            pca = PCA(n_components=len(latents))
            final_components = pca.fit_transform(concatenated_latents)
                
    merged_latents = pd.DataFrame(final_components, columns=latents)
    merged_latents['IID'] = IID
    merged_latents['FID'] = FID
        
    cols = ['FID', 'IID'] + [col for col in merged_latents.columns if col not in ['FID', 'IID']]
    merged_latents = merged_latents[cols]
    
    merged_latents.to_csv(f"{args.path4res}/merged_latents_raw.tsv", sep="\t", index=False)