import pandas as pd
import re
def clean_sp2(value):
    value = re.sub(r'\(.*?\)', '', value)
    value = value.replace('NONE', '')
    return value

def summarise_singlerun(path_toploci, path_summary):
    with open(path_summary, "w") as f: 
        merged_toploci = pd.read_table(path_toploci)
        f.write(f"Total number of Toploci in the merged file: {len(merged_toploci)}")
        f.write(f"Number of Toploci with SP2: {len(merged_toploci[merged_toploci['SP2'] != 'NONE'])}")

        merged_toploci['SP2'] = merged_toploci['SP2'].apply(clean_sp2)
        merged_toploci['SP2'] = merged_toploci['SP2'].str.split(',')
        merged_toploci = merged_toploci.explode('SP2')    
        merged_toploci = merged_toploci[merged_toploci['SP2'] != '']
        f.write(f"Total number of unique SNPs : {len(set(merged_toploci.SNP).union(set(merged_toploci.SP2)))}")

def summarise_multirun(path_merged_toploci, path_summary, common_run_tag=""):
    with open(path_summary, "w") as f:    
        merged_toploci = pd.read_table(path_merged_toploci)
        f.write(f"Total number of Toploci in the merged file: {len(merged_toploci)}\n")
        f.write(f"Number of Toploci with SP2: {len(merged_toploci[merged_toploci['SP2'] != 'NONE'])}\n")
        
        if common_run_tag:
            merged_toploci.Run = merged_toploci.Run.str.replace(f"{common_run_tag}_", "").str.replace(",","+")
            merged_toploci.RunPheno = merged_toploci.RunPheno.str.replace(f"{common_run_tag}_", "").str.replace(",","+")
            path_cleaner = path_merged_toploci.replace(f".{path_merged_toploci.split('.')[-1]}", f"_cleaner.{path_merged_toploci.split('.')[-1]}")
            merged_toploci.to_csv(path_cleaner, sep="\t", index=False)
        else:
            merged_toploci.Run = merged_toploci.Run.str.replace(",","+")

        f.write("--------------------------------------------------------\n")

        combos = set(merged_toploci.Run)
        unique_runs = [r for r in combos if "+" not in r]
        combo_runs = [r for r in combos if "+" in r]

        str_unique_toploci = "Unique Toploci → "
        str_unique_toploci_withSP2 = "Unique Toploci with SP2 → "
        for run in unique_runs:
            subset = merged_toploci[merged_toploci.Run == run]
            str_unique_toploci += f"{run}: {len(subset)} | "
            str_unique_toploci_withSP2 += f"{run}: {len(subset[subset['SP2'] != 'NONE'])} | "
        f.write(f"\n{str_unique_toploci}\n{str_unique_toploci_withSP2}\n")

        str_overlapping_toploci = "Overlapping Toploci → "
        str_overlapping_toploci_withSP2 = "Overlapping Toploci with SP2 → "
        for run in combo_runs:
            subset = merged_toploci[merged_toploci.Run == run]
            str_overlapping_toploci += f"{run}: {len(subset)} | "
            str_overlapping_toploci_withSP2 += f"{run}: {len(subset[subset['SP2'] != 'NONE'])} | "
        f.write(f"\n{str_overlapping_toploci}\n{str_overlapping_toploci_withSP2}\n")

        f.write("--------------------------------------------------------\n")

        merged_toploci['SP2'] = merged_toploci['SP2'].apply(clean_sp2)
        merged_toploci['SP2'] = merged_toploci['SP2'].str.split(',')
        merged_toploci = merged_toploci.explode('SP2')    
        merged_toploci = merged_toploci[merged_toploci['SP2'] != '']
        f.write(f"Total number of unique SNPs in the merged file : {len(set(merged_toploci.SNP).union(set(merged_toploci.SP2)))}\n")

        f.write("--------------------------------------------------------")

        str_unique_SNPs = "Unique SNPs → "
        for run in unique_runs:
            subset = merged_toploci[merged_toploci.Run == run].copy()
            subset['SP2'] = subset['SP2'].apply(clean_sp2)
            subset['SP2'] = subset['SP2'].str.split(',')
            subset = subset.explode('SP2')    
            subset = subset[subset['SP2'] != '']
            str_unique_SNPs += f"{run}: {len(set(subset.SNP).union(set(subset.SP2)))} | "
        f.write(f"\n{str_unique_SNPs}\n")

        str_overlapping_SNPs = "Overlapping SNPs → "
        for run in combo_runs:
            subset = merged_toploci[merged_toploci.Run == run].copy()
            subset['SP2'] = subset['SP2'].apply(clean_sp2)
            subset['SP2'] = subset['SP2'].str.split(',')
            subset = subset.explode('SP2')    
            subset = subset[subset['SP2'] != '']
            str_overlapping_SNPs += f"{run}: {len(set(subset.SNP).union(set(subset.SP2)))} | "
        f.write(f"\n{str_overlapping_SNPs}\n")