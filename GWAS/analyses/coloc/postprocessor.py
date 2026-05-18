import os
import argparse
import logging
import re

import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import Counter

def clean_merged_model_table(df):
    def extract_fold(text):
        match = re.search(r'fold(\d+)_', text)
        return match.group(1) if match else None

    def extract_seed(text):
        match = re.search(r'_seed(\d+)_', text)
        if match:
            return match.group(1)
        else:
            match = re.search(r'seed_(\d+)', text)
            return match.group(1) if match else '1701'

    df['fold_ref'] = df['tID_ref'].apply(extract_fold)
    df['fold_best_match'] = df['tID_best_match'].apply(extract_fold)
    df['seed_ref'] = df['tID_ref'].apply(extract_seed)
    df['seed_best_match'] = df['tID_best_match'].apply(extract_seed)

    return df

def flatten_mappings_dict(d):
    items = {}
    def recurse(d, parent_key):
        combined_leaf_values = []
        for k, v in d.items():
            if isinstance(v, dict):
                recurse(v, k)
                combined_leaf_values.extend([value for value in v.values() if not isinstance(value, dict)])
            else:
                items[k] = v 
                combined_leaf_values.append(v)  
        if parent_key:  
            items[parent_key] = ','.join(combined_leaf_values)  
    recurse(d, '')
    return items

def get_counts(mappings, df_colocs):
    key_row_counts = {key: 0 for key in mappings}  # Task 1
    key_value_counts = {key: 0 for key in mappings}  # Task 2
    
    # Task 3: initialise value_counts with all possible values from mappings
    all_values = {value for values in mappings.values() for value in values.split(',')}
    value_counts = {value: 0 for value in all_values}

    mappings_sets = {key: set(values.split(',')) for key, values in mappings.items()}

    for _, row in df_colocs.iterrows():
        traits_set = set(row['unique_traits'].split(','))

        for key, values_set in mappings_sets.items():
            if values_set.intersection(traits_set):
                key_row_counts[key] += 1  # Task 1
                key_value_counts[key] += len(values_set.intersection(traits_set))  # Task 2

        # Task 3
        for trait in traits_set:
            if trait in value_counts:
                value_counts[trait] += 1

    return key_row_counts, key_value_counts, value_counts

def is_substring_in_latents(trait, latents):
    return any(latent in trait for latent in latents)

def process_final_locus(pth_final_locus, selected_latents=None):  
    try:
        df_final_locus = pd.read_csv(pth_final_locus, sep="\t")
        locusID = re.search(r'locus_(\d+)_final_locus_table.tsv', os.path.basename(pth_final_locus))[1]

        if selected_latents is not None:
            final_locus_Zs = df_final_locus[df_final_locus['trait'].apply(lambda x: is_substring_in_latents(x, selected_latents))]
        else:
            final_locus_Zs = df_final_locus[df_final_locus.trait.str.match(r'(^|.*_)Z\d+(\.|$)')]
        if len(final_locus_Zs) == 0:
            logging.warning(f"Warning for {os.path.basename(pth_final_locus)}: No Zs found in traits. Skipping.")
            return None
        n_subloci = len(final_locus_Zs.sub_locus.unique())
        return {"locusID": locusID, "nIndSignals": n_subloci}
            
    except Exception as e:
        logging.error(f"Error (FinalLocus) in reading {os.path.basename(pth_final_locus)}. {e}")
        return None
        
def process_H4(pth_H4, selected_latents=None):   
    try:
        if not os.path.isfile(pth_H4):
            logging.warning(f"Warning for {os.path.basename(pth_H4)}: H4 file not found. Skipping.")
            return []
            
        df_H4 = pd.read_csv(pth_H4, sep="\t")
        locusID = re.search(r'locus_(\d+)_colocalization.table.H4.tsv', os.path.basename(pth_H4))[1]
            
        res = []
        
        if selected_latents is not None:
            sig_Zs_with_traits = df_H4[(df_H4['t1'].apply(lambda x: is_substring_in_latents(x, selected_latents)) != df_H4['t2'].apply(lambda x: is_substring_in_latents(x, selected_latents)))]
        else:
            sig_Zs_with_traits = df_H4[(df_H4['t1'].str.match(r'(^|.*_)Z\d+(\.|$)') != df_H4['t2'].str.match(r'(^|.*_)Z\d+(\.|$)'))]
        if len(sig_Zs_with_traits) == 0:
            logging.warning(f"H4 Warning for {os.path.basename(pth_H4)}: No Zs found in traits. Skipping H4.")
            return res
        
        sig_unique_subloci = sig_Zs_with_traits.g1.unique()
        for u_subloci in sig_unique_subloci:
            sig_subloci = sig_Zs_with_traits[sig_Zs_with_traits.g1 == u_subloci]
            uniques = list(set(list(sig_subloci.t1.unique()) + list(sig_subloci.t2.unique())))

            unique_Zs = ",".join([x.split(".")[0] for x in uniques if re.match(r'(^|.*_)Z\d+(\.|$)', x)]) 
            unique_traits = ",".join([x for x in uniques if not re.match(r'(^|.*_)Z\d+(\.|$)', x)])
            
            if len(unique_Zs) > 0 and len(unique_traits) > 0:
                res.append({
                    "locusID": locusID,
                    "sublocusID": u_subloci,
                    "unique_Zs": unique_Zs,
                    "unique_traits": unique_traits
                })           
        
        return res
            
    except Exception as e:
        logging.error(f"Error (H4) in reading {os.path.basename(pth_H4)}. {e}")
        return []

def process_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--res_path', action="store", default="", help="Path where the coloc results are available.")
    parser.add_argument('--out_path', action="store", default="", help="Path to store the results. Keep it blank to store the results inside the res_path folder or shortlisted_latents_path (if supplied).")
    parser.add_argument('--mappings', action="store", default="cardiac", help="Which mappings to use (the intial key inside the mappings.json file).")
    
    parser.add_argument('--shortlisted_latents_path', action="store", default="", help="Path where the results for the shorlisted latents are present. Leave it blank to skip")    

    args, unknown_args = parser.parse_known_args()

    return args, unknown_args

def main():
    args, unknown_args = process_arguments()
    
    if bool(args.shortlisted_latents_path):
        if not bool(args.out_path):
            args.out_path = args.shortlisted_latents_path
        if os.path.isfile(f"{args.shortlisted_latents_path}/latents_similarity_filtered_clean.tsv"):
            shortlisted_latents = pd.read_table(f"{args.shortlisted_latents_path}/latents_similarity_filtered_clean.tsv")
        else:
            shortlisted_latents = pd.read_table(f"{args.shortlisted_latents_path}/latents_similarity_filtered.tsv")
            shortlisted_latents = clean_merged_model_table(shortlisted_latents)
            shortlisted_latents.to_csv(f"{args.shortlisted_latents_path}/latents_similarity_filtered_clean.tsv", sep="\t", index=False)
        refs = "S"+shortlisted_latents.seed_ref.astype(str) + "_F" + shortlisted_latents.fold_ref.astype(str) + "_" + shortlisted_latents.col_ref.astype(str)
        best_matches = "S"+shortlisted_latents.seed_best_match.astype(str) + "_F" + shortlisted_latents.fold_best_match.astype(str) + "_" + shortlisted_latents.col_best_match.astype(str)
        selected_latents = list(set(refs).union(set(best_matches)))
    else:
        selected_latents = None

    if not bool(args.out_path):
        args.out_path = args.res_path
    else:
        os.makedirs(args.out_path, exist_ok=True)

    coloc_name = os.path.basename(args.res_path)
    logging.basicConfig(filename=f"{args.out_path}/log_{coloc_name}.log", level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info(f"Processing {coloc_name}........................................................................")

    pth_final_locus_tabs = glob(f"{args.res_path}/results/locus_*_final_locus_table.tsv")

    ind_sig_counts = []
    colocs = []
    for pth_final_locus_tab in tqdm(pth_final_locus_tabs):
        ind_sig_counts.append(process_final_locus(pth_final_locus=pth_final_locus_tab, selected_latents=selected_latents))
        pth_H4 = pth_final_locus_tab.replace("_final_locus_table.tsv", "_colocalization.table.H4.tsv")
        colocs += process_H4(pth_H4, selected_latents=selected_latents)

    ind_sig_counts = [x for x in ind_sig_counts if x is not None]
    df_ind_sig = pd.DataFrame(ind_sig_counts)
    df_ind_sig.to_csv(f"{args.out_path}/indsigs_{coloc_name}.tsv", sep="\t", index=False)

    df_colocs = pd.DataFrame(colocs)
    df_colocs.to_csv(f"{args.out_path}/colocs_{coloc_name}.tsv", sep="\t", index=False)

    with open(f"{os.path.dirname(os.path.realpath(__file__))}/mappings.json", "r") as f:
        mappings = flatten_mappings_dict(json.load(f)[args.mappings])
    key_row_counts, key_value_counts, value_counts = get_counts(mappings, df_colocs)

    with open(f"{args.out_path}/coloc_counts_{coloc_name}.txt", 'w') as f:
        f.write("Number of independent signals within loci (latents):----------------------\n")
        if len(df_ind_sig) == 0:
            f.write(f"0\n")
        else:
            f.write(f"{df_ind_sig.nIndSignals.sum()}\n")
 
        f.write("Total number of colocalisation signals:----------------------\n")
        f.write(f"{len(df_colocs)}\n")
 
        f.write("------------------------------------------------------------------\n")
        
        f.write("Number of sub-loci for each category:----------------------\n")
        for key, count in key_row_counts.items():
            if count > 0:
                f.write(f"{key}: {count}\n")
        
        f.write("------------------------------------------------------------------\n")
        f.write("\nNumber of appearances of each unique trait:----------------------\n")
        for value, count in value_counts.items():
            if count > 0:
                f.write(f"{value}: {count}\n")  
        
        f.write("------------------------------------------------------------------\n")
        f.write("\nNumber of appearances of each category:----------------------\n")
        for value, count in key_value_counts.items():
            if count > 0:
                f.write(f"{value}: {count}\n")

if __name__ == "__main__":
    main()