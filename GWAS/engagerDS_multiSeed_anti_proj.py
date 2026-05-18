import os
import sys
import re
import numpy as np
import argparse
from string import Template
import pandas as pd
import getpass
from copy import deepcopy
from glob import glob
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, os.getcwd()) #to handle the sub-foldered structure of the tricoder

from utils.python_utils import process_unknown_args, DotDict
from H5tools.traverse_embH5 import process_embs as embsH5_to_npy
from GWAS.prep_latents import finalise_latentDFs
from GWAS.create_conf import ConfFile
import yaml

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--selection_tsv', default="/group/glastonbury/soumick/GWAS/shortlisted_latents_models/posthoc_rscore/5Seeds_f0_DiffAEFP16_128_Crop3D/thres_gt80p/latents_similarity_filtered.tsv", type=str, help='Path to the tsv file containing the shortlisted latents.')    
    
    parser.add_argument('-ph', '--path_phenotypes', type=str, default="", help='Path to the phenotype file. [Default: Blank >> They will be generated using the following parameters.]') #If the phenotype file is available, that can directly be provided
    
    parser.add_argument('-dsv', '--dsV', type=str, default="V3", help='Version of the dataset to be used [This also is the value of ~ in subsequent arguments]. [Default: V2. For V1, leave it blank.]')
    
    parser.add_argument('-n', '--projectName', type=str, default="DAE128_5Sd_nr80_discov_splprj", help='To be used as the project name for the GWAS and other processed data. [Default: "" (trainID is being used)]')    
    parser.add_argument('-o', '--out_path', type=str, default="/group/glastonbury/GWAS/F20208v3_DiffAE/nonselect_latents_r80", help='Path to store the output files. [Default: Left blank >> If path_phenotypes is provided -> same folder as the path_phenotypes. If not -> Same as the model folder, with GWAS_fullDS~ subfolder]')
    parser.add_argument('-os', '--out_subdir_tag', type=str, default="", help='Tag to be used for the output subfolder along with (GWAS_), only if out_path is blank.')
    parser.add_argument('-p', '--pipeline_root', type=str, default="/group/glastonbury/soumick/codebase", help='Path to the root of the pipeline. Leave it blank if the pipeline is in the same folder as the directory from where the current script is being launched.')
    parser.add_argument('-w', '--custom_work_dir', action=argparse.BooleanOptionalAction, default=True, help='If custom folder with the projectName is to be created inside the default work dir: /scratch/$USER/nf-gwas-work. If blank, no subdir will be created. [Default: Blank]')
    parser.add_argument("--email4slurm", type=str, default="soumick.chatterjee@fht.org", help="Email ID to be used for SLURM notifications.")
    parser.add_argument("--nodes2exclude", type=str, default="", help="Coma-separated list of nodes to exclude.")

    parser.add_argument('-lg', '--launch_gwas', action=argparse.BooleanOptionalAction, default=False, help='Whether to launch the GWAS or not. [Default: True] If only to process the embeddings, we can set it to False.')
    parser.add_argument('-spm', '--split_proj_maxPh', type=int, default=100, help='Split the phenotypes into subsets as multi-projects, each with a maximum of this many phenotypes.')
    
    #The following parameters will only be used if the phenotype file is not provided
    parser.add_argument('-nl', '--nLatents', type=int, default=128, help='Number of latent factors. [Default: 128]')
    parser.add_argument('-sl', "--selectiveLatents", type=str, default=None, help="Comma separated list of latent factors to be used for the GWAS. [Default: None]")
    parser.add_argument('-i', '--in_root_path', type=str, default="/project/ukbblatent/Out/Results", help='Path to the model folder (containing a folder with the trainID).')
    parser.add_argument('-is', '--in_train_subfold', type=str, default="", help='Subfolder inside the model (trainID) folder containing the embeddings. [Default: "" (Output_fullDS~)]')
    parser.add_argument('-cd', '--clinical_data_path', type=str, default="/project/ukbblatent/clinicaldata", help='Path where there are the clinical data TSVs (to be used for filters).')

    parser.add_argument("--subIDs", help="Coma-seperated list of subject IDs. Blank for all.", default="")
    parser.add_argument("--txt_subIDs", help="Path to a text file containing coma-seperate list of subIDs (This has priority over subIDs)", default="/group/glastonbury/GWAS/F20208v3_DiffAE/subIDs_MRI_discovery_V2WBRIT.txt")
    parser.add_argument("--filter_gensex", type=int, default=0, help="Filter based on genetic sex. [0 (default): No filter, 1: Female, 2: Male]")
    parser.add_argument("--filter_genethnicity", action=argparse.BooleanOptionalAction, help="Whether to filter the subjects based on genetic thenicity [True implies only white british will be included]", default=False)
    parser.add_argument("--filter_flip", action=argparse.BooleanOptionalAction, help="Flip the genetic ethnicity filter", default=False)

    parser.add_argument("--data_tags", help="Coma-seperated list of data tags. Blank for all.", default="primary_LAX_4Ch_transverse_0")
    parser.add_argument("--flatten_embs", action=argparse.BooleanOptionalAction, help="Whether to flatten the data before storing or not.", default=False)
    parser.add_argument("--quantile_transform", action=argparse.BooleanOptionalAction, help="Whether to apply quantile transformer before preparing the final dataframe.", default=True)
    
    parser.add_argument("-cov", "--covariates", type=str, help="Path to the covariates file", default="/group/glastonbury/GWAS/inputs/covariates/cov_newset_chp_F20208_Long_axis_heart_images_DICOM_H5v3.tsv")
    parser.add_argument("-ns", "--no_noise", action=argparse.BooleanOptionalAction, default=True, help='Whether to avoid noisy images (using the "_NOnoise" version of the covariates file)')
    
    parser.add_argument("--cov_cols", type=str, help="Coma-seperated list of covarites columns", default="Genotype_Batch,Sex,MRI_Date,MRI_Centre,MRI_Visit,Age,BSA")
    parser.add_argument("--cov_cat_cols", type=str, help="Coma-seperated list of categorical covarites columns", default="Genotype_Batch,Sex,MRI_Date,MRI_Centre,MRI_Visit")
    parser.add_argument("--max_cat_levels", type=int, help="Maximum number of categorical values any categorical column might have.", default=10)
    
    parser.add_argument("--gwasfilt_info", type=float, help="INFO filter for the GWAS (will be used as the value for regenie_min_imputation_score)", default=0.3)
    return parser

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

def parse_selection_tsv(selection_tsv):
    df = pd.read_table(selection_tsv)
    if not selection_tsv.endswith("_clean.tsv"):
        df = clean_merged_model_table(df)
        df.to_csv(selection_tsv.replace(".tsv", "_clean.tsv"), sep="\t", index=False)
    
    data = {
        "tID": pd.concat([df['tID_ref'], df['tID_best_match']]),
        "seed": pd.concat([df['seed_ref'], df['seed_best_match']]),
        "fold": pd.concat([df['fold_ref'], df['fold_best_match']]),
        "col": pd.concat([df['col_ref'], df['col_best_match']]),
    }
    df = pd.DataFrame(data)
    
    unique_seed = len(df['seed'].unique()) == 1
    unique_fold = len(df['fold'].unique()) == 1
    
    def create_tag(row):
        parts = []
        if not unique_seed:
            parts.append(f"S{row['seed']}")
        if not unique_fold:
            parts.append(f"F{row['fold']}")
        parts.append(row['col'] if "Z" in row['col'] else f"Z{row['col']}")
        return '_'.join(parts)

    df['latent'] = df.apply(create_tag, axis=1)
    df['tag'] = df['latent'].str.extract(r'^(.*?)(?=_Z)')
    return df[['tag', 'tID', 'latent']].rename(columns={'tID': 'trainID'}).drop_duplicates()

def cov_sex_removal(cov_cols):
    cov_cols = cov_cols.split(",") if isinstance(cov_cols, str) else cov_cols
    filter_list = ['Gender', 'Sex', 'Genetic_sex', 'Biological_sex']
    lower_filter_list = set(item.lower() for item in filter_list)
    cov_cols = [item for item in cov_cols if item.lower() not in lower_filter_list]
    return ','.join(cov_cols)
    
def filter_subs(args):
    filtered_subs = []
    flag = False
    if args['txt_subIDs'] != "":
        with open(args['txt_subIDs'], 'r') as file:
            filtered_subs.append(list(map(int, file.read().split(","))))
            flag = True
    elif args['subIDs'] != "":
        filtered_subs.append(list(map(int, args['subIDs'].split(","))))
        flag = True
    if args['filter_gensex']:
        args['cov_cols'] = cov_sex_removal(args['cov_cols'])
        args['cov_cat_cols'] = cov_sex_removal(args['cov_cat_cols'])
        genetic_sex = pd.read_table(f"{args['clinical_data_path']}/genetic_sex_chip.tsv").set_index('f.eid')
        if args['filter_gensex'] == 1:
            filtered_subs.append(list(genetic_sex[genetic_sex['f.22001.0.0']==0.0].index)) #female
        else:
            filtered_subs.append(list(genetic_sex[genetic_sex['f.22001.0.0']==1.0].index)) #male
        flag = True
    if args['filter_genethnicity']:
        genetic_ethnicity = pd.read_table(f"{args['clinical_data_path']}/genetic_ethnicity.tsv").set_index('f.eid')
        if args['filter_flip']:
            filtered_subs.append(list(genetic_ethnicity[genetic_ethnicity.isnull().any(1)].index))
        else:
            filtered_subs.append(list(genetic_ethnicity.dropna().index))
        flag = True
    if len(filtered_subs) == 1:
        args['subIDs'] = ','.join(map(str, filtered_subs[0]))
    elif len(filtered_subs) > 1:
        args['subIDs'] = ','.join(map(str, set.intersection(*map(set, filtered_subs))))
    if flag and args['subIDs'] == "":
        sys.exit("No subjects left after filtering. Exiting...")


def prep_args_processEmbs(args, trainID="", save_tag="", out_path=""):
    trainID = args['trainID'] if trainID == "" else trainID
    args_processEmbs = DotDict()
    args_processEmbs.in_path = f"{args['in_root_path']}/{trainID}/{args['in_train_subfold']}/emb.h5"
    args_processEmbs.out_path = out_path if bool(out_path) else args['out_path']
    args_processEmbs.subIDs = args['subIDs']
    args_processEmbs.data_tags = args['data_tags']
    args_processEmbs.flatten_data = args['flatten_embs']
    args_processEmbs.save_csv = False
    args_processEmbs.prep_Zs = False
    args_processEmbs.save_tag = save_tag
    return args_processEmbs

def merge_select_embs(in_path, out_path="", selected_latents=None):
    out_path = in_path if out_path == "" else out_path
    pth_embs = glob(f"{in_path}/*processed.tsv")
    embs = []
    for pth_emb in pth_embs:
        emb = pd.read_table(pth_emb)
        tag = os.path.basename(pth_emb).replace("processed.tsv", "")
        cols = {col: f"{tag}{col}" if re.search(r'Z\d+$', col) else col for col in emb.columns}
        emb.rename(columns=cols, inplace=True)
        embs.append(emb)
    df = pd.concat(embs, axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    if bool(selected_latents):
        df.to_csv(f"{out_path}/processed_preselection.tsv", sep='\t', index=False)
        # nonZ_cols = [col for col in df.columns if not re.search(r'Z\d+$', col)]
        # df = df[nonZ_cols + selected_latents]
        non_selected_cols = [col for col in df.columns if col not in selected_latents]
        args['selectiveLatents'] = ','.join([col for col in non_selected_cols if re.search(r'Z\d+$', col)])
        df = df[non_selected_cols]
        df.to_csv(f"{out_path}/processed.tsv", sep='\t', index=False)
    else:
        df.to_csv(f"{out_path}/processed.tsv", sep='\t', index=False)

def prep_args_conf(args, args4conf):
    if bool(args['selectiveLatents']):
        latents = args['selectiveLatents']
    else:
        for i in range(args["nLatents"]):
            if i == 0:
                latents = f"Z{i}"
            else:
                latents += f",Z{i}"
    
    args4conf['params'] = {
        "project": args['projectName'],
        "phenotypes_filename": args['path_phenotypes'] if bool(args['path_phenotypes']) else f"{args['out_path']}/processed.tsv",
        "phenotypes_columns": latents,
        "covariates_filename": args['covariates'],
        "covariates_columns": args['cov_cols'],
        "covariates_cat_columns": args['cov_cat_cols'],
        "maxCatLevels": args['max_cat_levels'],
        "regenie_min_imputation_score": args['gwasfilt_info'],
    }

    return args4conf

def prep_args_conf_splitproj(args, args4conf):
    if bool(args['selectiveLatents']):
        latents = args['selectiveLatents']
    else:
        for i in range(args["nLatents"]):
            if i == 0:
                latents = f"Z{i}"
            else:
                latents += f",Z{i}"
                
    latents = latents.split(",")
    split_latents = [latents[i:i + args['split_proj_maxPh']] for i in range(0, len(latents), args['split_proj_maxPh'])]
    if len(split_latents) == 1:
        return prep_args_conf(args, args4conf)
    
    project_tab = []
    for i, latents in enumerate(split_latents):
        project_tab.append({
            "project_id": f"{args['projectName']}_part{i}",
            "pheno_file": args['path_phenotypes'] if bool(args['path_phenotypes']) else f"{args['out_path']}/processed.tsv",
            "pheno_cols": ",".join(latents),
            "pheno_binary": "False",
            "pheno_model": "additive",
            "cov_file": args['covariates'],
            "cov_cols": args['cov_cols'],
            "cov_cat_cols": args['cov_cat_cols'],
        })
    project_tab = pd.DataFrame(project_tab)
    project_tab.to_csv(f"{args['out_path']}/project_table.tsv", sep="\t", index=False)
    args['part_proj_IDs'] = ";".join(project_tab.project_id)
    args['part_proj_phenos'] = ";".join(project_tab.pheno_cols)
    
    args4conf['params'] = {
        "project": args['projectName'],
        "projects_table": f"{args['out_path']}/project_table.tsv",
        "maxCatLevels": args['max_cat_levels'],
        "regenie_min_imputation_score": args['gwasfilt_info'],
    }

    return args4conf

def create_conf(args, args4conf, tag="", splitproj=False):
    conf = ConfFile("GWAS/template.conf")

    for key, value in args4conf.items():
        if "§" in key:
            key_parts = key.split("§")
            conf.set_value(section=key_parts[0], key=key_parts[1], value=value)
        elif isinstance(value, dict):
            for _k, _v in value.items():
                conf.set_value(section=key, key=_k, value=_v)
        else:
            conf.set_value(section=key, value=value)
            
    if splitproj:
        params2remove = ["phenotypes_filename", "phenotypes_columns", "phenotypes_binary_trait", "covariates_filename", "covariates_columns", "covariates_cat_columns"]
        for param in params2remove:
            conf.remove_key(section="params", key=param)

    conf.save(f"{args['out_path']}/run_gwas{tag}.conf")
    print("Conf file created!")

def create_sbatch(args, phenotypes="", tag="", splitproj=False):
    substitutions = {
        '§§JOBNAME§§': args['projectName'],       
        '§§EMAIL§§': args['email4slurm'],  
        '§§WORKDIR§§': args['out_path'],
        '§§PIPELINEROOT§§': args['pipeline_root'],
        '§§CONFFILE§§': f"{args['out_path']}/run_gwas{tag}.conf",  
        '§§PHENOLIST§§': phenotypes,
    }

    if args['nodes2exclude'] != "":
        substitutions['§§NF_NODEEXCLD§§'] = f"-process.clusterOptions=--exclude=\'{args['nodes2exclude']}\'"
        substitutions['§§NODEEXCLUDE§§'] = f"--exclude={args['nodes2exclude']}"
    else:
        substitutions['§§NF_NODEEXCLD§§'] = ""
        substitutions['§§NODEEXCLUDE§§'] = ""
        
    if splitproj:
        substitutions['§§PARTJOBS§§'] = args['part_proj_IDs']
        substitutions['§§PARTPHENOLIST§§'] = args['part_proj_phenos']
        with open('GWAS/template_sbatch_multi.sh', 'r') as file:
            content = file.read()
    else:
        with open('GWAS/template_sbatch.sh', 'r') as file:
            content = file.read()

    for key, value in substitutions.items():
        content = content.replace(key, value)

    with open(f"{args['out_path']}/run_gwas{tag}.sh", 'w') as file:
        file.write(content)

    print("SLURM script created!")

if __name__ == "__main__":
    parser = getARGSParser()
    args, args4conf = parser.parse_known_args() #unknown configs are only used for the creating for the conf file
    args4conf = process_unknown_args(args4conf)
    args = vars(args)
    
    if args['gwasfilt_info']:
        args['projectName'] = f"INF{int(args['gwasfilt_info']*100)}_{args['projectName']}"

    if args['filter_genethnicity'] and "WBRIT" not in args['projectName']:
        if args['filter_flip']:
            args['projectName'] = f"NonWBRIT_{args['projectName']}"
        else:
            args['projectName'] = f"WBRIT_{args['projectName']}"
            
    if args['filter_gensex']:
        tag_gensex = "fMl" if args['filter_gensex'] == 1 else "Ml"
        args['projectName'] = f"{tag_gensex}_{args['projectName']}"

    if args['quantile_transform'] and "Qntl" not in args['projectName']:
        args['projectName'] = f"Qntl_{args['projectName']}"
        
    if args['no_noise']:
        args['covariates'] = args['covariates'].replace(".tsv", "_NOnoise.tsv") if "NOnoise" not in args['covariates'] else args['covariates']
        args['projectName'] = f"nNs_{args['projectName']}"
        
    if bool(args['dsV']):
        args['projectName'] = f"{args['projectName']}_fullDS{args['dsV']}" if args['dsV'].lower() not in args['projectName'].lower() else args['projectName']
        dsV_cov = args['dsV'].replace("Only", "").lower() #If we are running V3Only dataset, we can still use the covariates file for V3 dataset.
        if dsV_cov not in args['covariates'].lower():
            print(f"WARNING: DS version {dsV_cov} not found in the name of the covariates file! Assuming the covariates file is the same as the one used for the dataset (v2), v2 will be added to the name...")
            args['covariates'] = args['covariates'].replace("_NOnoise.tsv", f"{dsV_cov}_NOnoise.tsv") if args['no_noise'] else args['covariates'].replace(".tsv", f"{dsV_cov}.tsv")

    if args['out_subdir_tag'] == "":
        args['out_subdir_tag'] = args['projectName']
        
    if bool(args['path_phenotypes']):
        if args['out_path'] == "":
            args['out_path'] = f"{os.path.dirname(args['path_phenotypes'])}/GWAS"
            if args['out_subdir_tag'] != "":
                args['out_path'] = f"{args['out_path']}_{args['out_subdir_tag']}"
    else:
        if args['in_train_subfold'] == "":
            args['in_train_subfold'] = f"Output_fullDS{args['dsV']}"

        out_latent_store = f"{args['out_path']}/latents"
        if args['out_subdir_tag'] != "":
            args['out_path'] = f"{args['out_path']}/{args['out_subdir_tag']}"
        else:
            args['out_path'] = f"{args['out_path']}/GWAS_fullDS{args['dsV']}"
    os.makedirs(args['out_path'], exist_ok=True)

    if args['pipeline_root'] != "" and args['pipeline_root'][-1] != "/":
        args['pipeline_root'] = args['pipeline_root']+"/"

    if args['custom_work_dir']:
        args4conf['workDir'] = f"/scratch/{getpass.getuser()}/nfgws/{args['projectName']}"
        os.makedirs(args4conf['workDir'], exist_ok=True)

    #filter subjects [if needed]
    filter_subs(args)

    if args['path_phenotypes']:
        sys.exit("Phenotype file supplied! But it's not yet implemented...")
        print("Phenotype file supplied! Skipping the processing of the emb.h5, rather processing the tsv file directly...")
        if "_raw" in args['path_phenotypes']:
            print("Raw phenotype file detected! Processing...")
            df = pd.read_table(args['path_phenotypes'], index_col="FID") if args['path_phenotypes'].endswith(".tsv") else pd.read_csv(args['path_phenotypes'], index_col="FID")
            df = df.loc[df.index.intersection(list(map(int, args['subIDs'].split(","))))]
            if args['quantile_transform']:
                for col in df.columns:
                    if col in ["FID", "IID"]:
                        continue
                    normaliser = QuantileTransformer(output_distribution='normal')
                    df[col] = normaliser.fit_transform(np.expand_dims(df[col], axis=1)).squeeze()
            df.to_csv(args['path_phenotypes'].replace("_raw", ""), sep="\t")
            args['path_phenotypes'] = args['path_phenotypes'].replace("_raw", "")
        else:
            print("Processed phenotype file detected! Skipping the processing step...")
        gwas_ready = True
    else:
        # 
        selected_latents = parse_selection_tsv(args['selection_tsv'])
        args['selectiveLatents'] = ','.join(selected_latents['latent'].unique()) if not bool(args['selectiveLatents']) else args['selectiveLatents']
        
        # Processing the individual embeddings and saving them as an npy file
        unique_trainings = selected_latents[['tag', 'trainID']].drop_duplicates()
        os.makedirs(f"{args['out_path']}/latents_collect", exist_ok=True)
        for _, row in unique_trainings.iterrows():
            embsH5_to_npy(prep_args_processEmbs(args, trainID=row['trainID'], save_tag=f"{row['tag']}_", out_path=f"{args['out_path']}/latents_collect"))
            finalise_latentDFs(f"{args['out_path']}/latents_collect/{row['tag']}_emb.npy", f"{args['out_path']}/latents_collect/{row['tag']}_processed.tsv", quantile_transform=args['quantile_transform'])

        merge_select_embs(in_path=f"{args['out_path']}/latents_collect", out_path=args['out_path'], selected_latents=list(selected_latents.latent.unique()))
        args['path_phenotypes'] = f"{args['out_path']}/processed.tsv"

        #create conf file for GWAS pipeline
        if args['split_proj_maxPh'] in [0, -1]:
            args4conf = prep_args_conf(args, args4conf) 
            create_conf(args, args4conf)
            create_sbatch(args, phenotypes="") #TODO: fix it!
        else:
            args4conf = prep_args_conf_splitproj(args, args4conf)
            create_conf(args, args4conf, splitproj=True)
            create_sbatch(args, splitproj=True) 

        #create SLURM script and launch 
        if args['launch_gwas']:
            os.system(f"sbatch {args['out_path']}/run_gwas.sh")
            print("SLURM script sbatch-ed! Terminating GWAS engager...")