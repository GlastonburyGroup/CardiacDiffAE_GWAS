import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm
from collections import defaultdict
from glob import glob

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #TODO: find a better way to do this!
sys.path.insert(0, os.getcwd())

from GWAS.analyses.utils.toploci_merger import adjust_merged_loci
from GWAS.analyses.utils.summarise import summarise_multirun
from GWAS.analyses.utils.nlp import gen_wordcloud
from GWAS.analyses.tag_each_ldtrait import get_counts

def mahalanobis_distance(x, y, epsilon=1e-5):    
    data = np.array([x, y])
    variance = np.var(data, axis=0, ddof=1)    
    variance += epsilon
    squared_diff = (x - y)**2
    mahalanobis_squared = squared_diff / variance
    md = np.sqrt(np.sum(mahalanobis_squared))
    max_diff = (np.max(data, axis=0) - np.min(data, axis=0))**2
    max_mahalanobis_squared = max_diff / variance
    max_md = np.sqrt(np.sum(max_mahalanobis_squared))
    return md / (max_md + epsilon)

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trainIDs', type=str, default="F20208_heart_1Ses_time2slc_MskCrop128_V2_3D100ep_L1_4ChTrans128fold0_prec32_pythaemodel-vae,\
                        F20208_heart_1Ses_time2slc_MskCrop128_V2_3D100ep_seed1993_L1_4ChTrans128fold0_prec32_pythaemodel-vae,\
                        F20208_heart_1Ses_time2slc_MskCrop128_V2_3D100ep_seed42_L1_4ChTrans128fold0_prec32_pythaemodel-vae,\
                        F20208_heart_1Ses_time2slc_MskCrop128_V2_3D100ep_seed1994_L1_4ChTrans128fold0_prec32_pythaemodel-vae,\
                        F20208_heart_1Ses_time2slc_MskCrop128_V2_3D100ep_seed2023_L1_4ChTrans128fold0_prec32_pythaemodel-vae", help='Coma-seperated list of trainIDs to merge')
    parser.add_argument('--shortcodes', type=str, default="1701,1993,42,1994,2023", help='Coma-seperated list of shortcodes for each trainID')
    
    parser.add_argument("--train_root", type=str, default="../Out/Results", help="Root folder where the trainings are stored")
    parser.add_argument("--res_root", type=str, default="", help="Root folder where the results are stored (Only if train_root is not provided). Will ignore output_folder params. [Not implemented yet]")
    
    parser.add_argument("--gwas_folder", type=str, default="GWAS_fullDSV2", help="Folder inside train_root/trainID where the processed_raw.tsv is available (output of GWAS engager).")
    
    parser.add_argument('--path4res', type=str, default="../shortlisted_latents_models/posthoc", help='Path where the merged stuff will be saved.')
    parser.add_argument('--save_tag', type=str, default="5Seeds_VAE128_Crop3D_DSV2_f0", help='Tag to identify this merge')
    
    parser.add_argument('--similarity_measure', type=float, default=0, help='0: Cosine similarity, 1: Mahalanobis distance. 2: R-score. [Default: 0]')
    parser.add_argument('--thresh_similarity', type=float, default=0.5, help='Threshold for cosine similarity between two columns to be considered.')
    
    parser.add_argument('--save_individual_latents', type=int, default=0, help='Whether to save the individual latents or not. [Default: 0]')
    
    return parser

if __name__ == "__main__":
    args = getARGSParser().parse_args()

    args.trainIDs = [t.strip() for t in args.trainIDs.split(',')]
    args.shortcodes = args.shortcodes.split(',')
    
    match args.similarity_measure:
        case 0:
            args.path4res = f"{args.path4res}_cosine"
        case 1:
            args.path4res = f"{args.path4res}_mahalanobis"
        case 2:
            args.path4res = f"{args.path4res}_rscore"
        case _:
            raise NotImplementedError(f"Similarity measure ID {args.similarity_measure} is not implemented yet!")    
    
    args.path4res = f'{args.path4res}/{args.save_tag}/thres_gt{int(args.thresh_similarity*100)}p' 
    os.makedirs(args.path4res, exist_ok=True)

    latents = {
        trainID: pd.read_table(
            f'{args.train_root}/{trainID}/{args.gwas_folder}/processed_raw.tsv'
        )
        for trainID in tqdm(args.trainIDs)
    }

    column_similarities = []
    print("Calculating similarities between columns...")
    for trainID, reference_df in tqdm(latents.items()):
        for column in reference_df.columns[2:]:  # Assuming the first two columns are 'FID' and 'IID'
            i_trainID = args.trainIDs.index(trainID)
            reference_column_data = reference_df[column].values.reshape(-1)
            overall_best_similarity = -np.inf
            best_match_column = None
            best_tID = None

            for tID, df in latents.items():
                j_tID = args.trainIDs.index(tID)
                if i_trainID >= j_tID:
                    continue
                for other_column in df.columns[2:]:  # Again, assuming the first two columns are 'FID' and 'IID'
                    other_column_data = df[other_column].values.reshape(-1)
                    
                    match args.similarity_measure:
                        case 0:
                            similarity = 1 - distance.cosine(reference_column_data, other_column_data)
                        case 1:
                            # similarity = 1 - distance.mahalanobis(reference_column_data, other_column_data)
                            similarity = 1 - mahalanobis_distance(reference_column_data, other_column_data)
                        case 2:
                            similarity = np.corrcoef(reference_column_data, other_column_data)[0, 1]
                    similarity = abs(similarity)

                    if similarity > overall_best_similarity:
                        overall_best_similarity = similarity
                        best_match_column = other_column
                        best_tID = tID

            if overall_best_similarity != -np.inf:
                column_similarities.append({
                    'tID_ref': trainID,
                    'col_ref': column,
                    'tID_best_match': best_tID,
                    'col_best_match': best_match_column,
                    'best_similarity': overall_best_similarity
                })

    result_df = pd.DataFrame(column_similarities)
    result_df.to_csv(f'{args.path4res}/latents_similarity.tsv', sep='\t', index=False)
    filtered_df = result_df[result_df['best_similarity'] > args.thresh_similarity]
    filtered_df.to_csv(f'{args.path4res}/latents_similarity_filtered.tsv', sep='\t', index=False)

    print("Processing toplci.......")
    filtered_latents = defaultdict(dict)
    collect_toploci = []
    collect_ldtrait = []
    counts_str = ""
    counts = 0
    for i, trainID in tqdm(enumerate(args.trainIDs)):
        cols_ref = filtered_df[filtered_df['tID_ref'] == trainID]['col_ref'].values
        cols_best_match = filtered_df[filtered_df['tID_best_match'] == trainID]['col_best_match'].values
        cols = np.unique(np.concatenate([cols_ref, cols_best_match]))
        print(f'\nFor trainID {trainID}, {len(cols)} columns are selected.')
        counts_str += f'\nFor trainID {trainID}, {len(cols)} columns are selected.'
        counts += len(cols)

        colswithcodes = [f'{args.shortcodes[i]}_{col}' for col in cols]
        df = latents[trainID][['FID'] + list(cols)].copy()
        if args.save_individual_latents:
            df.to_csv(f'{args.path4res}/{trainID}_latents_filtered.tsv', sep='\t', index=False)
        df.set_index('FID', inplace=True)
        df.rename(columns=dict(zip(cols, colswithcodes)), inplace=True)
        filtered_latents[trainID] = df

        pth2toploci = glob(f'{args.train_root}/{trainID}/**/results/gwas/toploci/all_toploci_merged_raw.tsv', recursive=True)[0]
        toploci = pd.read_table(pth2toploci)
        toploci = toploci[toploci['Pheno'].isin(cols)]
        if args.save_individual_latents:
            df.to_csv(f'{args.path4res}/{trainID}_toploci_merged_filtered_raw.tsv', sep='\t', index=False)
        toploci['Pheno'] = toploci['Pheno'].apply(lambda x: f'{args.shortcodes[i]}_{x}')
        toploci['Run'] = trainID
        collect_toploci.append(toploci)

        if pth2ldtrait := glob(
            f'{args.train_root}/{trainID}/{args.gwas_folder}/**/results/analyses/LDLink_LDtraits/all.tagged.ldtrait.tsv',
            recursive=True,
        ):
            ldtrait = pd.read_table(pth2ldtrait[0])
            collect_ldtrait.append(ldtrait)
    
    counts_str += f'\n\nTotal number of columns selected: {counts}'
    with open(f'{args.path4res}/counts.txt', "w") as f:
        f.write(counts_str)

    merged_latents = pd.concat(filtered_latents.values(), axis=1)
    merged_latents['IID'] = merged_latents.index
    merged_latents.to_csv(f'{args.path4res}/merged_latents_filtered.tsv', sep='\t', index=False)

    merged_toploci = pd.concat(collect_toploci)
    merged_toploci.to_csv(f'{args.path4res}/merged_toploci_filtered_raw.tsv', sep='\t', index=False)

    merged_toploci = adjust_merged_loci(merged_toploci)
    merged_toploci['RunCount'] = merged_toploci['Run'].apply(lambda x: len(x.split(",")))
    merged_toploci.to_csv(f'{args.path4res}/merged_toploci_filtered.tsv', sep='\t', index=False)

    merged_toploci['Run'] = merged_toploci['Run'].apply(lambda x: ','.join([args.shortcodes[args.trainIDs.index(tID)] for tID in x.split(',')]))
    summarise_multirun(f'{args.path4res}/merged_toploci_filtered.tsv', f'{args.path4res}/merged_toploci_filtered.txt')
    
    print("Processing ldtraits.......")
    if len(collect_ldtrait) == len(args.trainIDs): #all the trainings already have the ldtraits
        print("All the trainings already have the ldtraits. Just processing them!")
        merged_ldtrait = pd.concat(collect_ldtrait)
        
        merged_toploci['SNPs'] = (merged_toploci['SNP'] + ',' + merged_toploci['SP2']).str.split(",").apply(lambda x: [s.split("_")[0] for s in x if s != "NONE"])
        
        selected_ldtraits = []
        for row in merged_toploci.itertuples():            
            ld = merged_ldtrait[merged_ldtrait['leadSNP'].apply(lambda x: x.split("_")[0]).isin(row.SNPs)].copy()
            ld['Pheno'] = row.Pheno
            ld['Run'] = row.Run
            selected_ldtraits.append(ld)
        
        df = pd.concat(selected_ldtraits)
        df = df.drop_duplicates(subset=['Query', 'GWAS Trait', 'RS Number', 'leadSNP'])
        
        df_sig_R5 = df[df["R2"] >= 0.5]
        df_sig_R6 = df[df["R2"] >= 0.6]
        df_sig_R8 = df[df["R2"] >= 0.8]
    
        if other_traits := df[df["tags"] == "NA"]["GWAS Trait"].unique().tolist():
            with open(f'{args.path4res}/all.ldtrait.other_traits.txt', "w") as f:
                f.write("Other traits, not in our list of OLSs, but significant LDLink_LDTrait:..........\n")
                f.write("\n".join(other_traits))

                if other_traits_sig_R5 := df_sig_R5[df_sig_R5["tags"] == "NA"]["GWAS Trait"].unique().tolist():
                    f.write("\n\n\n\nOther traits, as per LDLink_LDTrait, they are significant and the R2 is greater than or equals to 0.5:..........\n")
                    f.write("\n".join(other_traits_sig_R5))

                if other_traits_sig_R6 := df_sig_R6[df_sig_R6["tags"] == "NA"]["GWAS Trait"].unique().tolist():
                    f.write("\n\n\n\nOther traits, as per LDLink_LDTrait, they are significant and the R2 is greater than or equals to 0.6:..........\n")
                    f.write("\n".join(other_traits_sig_R6))

                if other_traits_sig_R8 := df_sig_R8[df_sig_R8["tags"] == "NA"]["GWAS Trait"].unique().tolist():
                    f.write("\n\n\n\nOther traits, as per LDLink_LDTrait, they are significant and the R2 is greater than or equals to 0.8:..........\n")
                    f.write("\n".join(other_traits_sig_R8))

            gen_wordcloud(df[df["tags"] == "NA"], df_is_indtraits=True, background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', filename=f"{args.path4res}/all.ldtrait.other_traits.png")

        get_counts(df=df, path=f"{args.path4res}/all.ldtrait.ols.tagcounts.txt", tag="# SNPs / Toploci with significant LDLink_LDTrait", mode='w')
        get_counts(df=df_sig_R5, path=f"{args.path4res}/all.ldtrait.ols.tagcounts.txt", tag="# SNPs / Toploci with significant LDLink_LDTrait and the R2 is greater than or equals to 0.5", mode='a')
        get_counts(df=df_sig_R6, path=f"{args.path4res}/all.ldtrait.ols.tagcounts.txt", tag="# SNPs / Toploci with significant LDLink_LDTrait and the R2 is greater than or equals to 0.6", mode='a')
        get_counts(df=df_sig_R8, path=f"{args.path4res}/all.ldtrait.ols.tagcounts.txt", tag="# SNPs / Toploci with significant LDLink_LDTrait and the R2 is greater than or equals to 0.8", mode='a')
    else:
        print("Some of the trainings do not have the ldtraits. So, performing ldtrait analysis from scratch!")
        print("Error: This is not implemented yet!") #TODO: Implement this!