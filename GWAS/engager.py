import os
import sys
import numpy as np
import argparse
from string import Template
import pandas as pd
import getpass
from copy import deepcopy
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, os.getcwd()) #to handle the sub-foldered structure of the tricoder

from utils.python_utils import process_unknown_args, DotDict
from H5tools.traverse_embH5 import process_embs as embsH5_to_npy
from GWAS.prep_latents import finalise_latentDFs
from GWAS.create_conf import ConfFile

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--trainID', type=str, help='Train ID (Name of the model). Will also be used as the project name for the GWAS (if projectName not specified).')
    parser.add_argument('-ph', '--path_phenotypes', type=str, default="", help='Path to the phenotype file. [Default: Blank >> They will be generated using the following parameters.]') #If the phenotype file is available, that can directly be provided
    
    parser.add_argument('-n', '--projectName', type=str, default="", help='To be used as the project name for the GWAS and other processed data. [Default: "" (trainID is being used)]')    
    parser.add_argument('-o', '--out_path', type=str, default="", help='Path to store the output files. [Default: Left blank >> If path_phenotypes is provided -> same folder as the path_phenotypes. If not -> Same as the model folder, with GWAS_fullDS subfolder]')
    parser.add_argument('-os', '--out_subdir_tag', type=str, default="", help='Tag to be used for the output subfolder along with (GWAS_), only if out_path is blank.')
    # parser.add_argument('-p', '--pipeline_root', type=str, default="/group/glastonbury/GWAS", help='Path to the root of the pipeline. Leave it blank if the pipeline is in the same folder as the directory from where the current script is being launched.')
    parser.add_argument('-p', '--pipeline_root', type=str, default="/group/glastonbury/soumick/codebase", help='Path to the root of the pipeline. Leave it blank if the pipeline is in the same folder as the directory from where the current script is being launched.')
    parser.add_argument('-w', '--custom_work_dir', action=argparse.BooleanOptionalAction, default=True, help='If custom folder with the projectName is to be created inside the default work dir: /scratch/$USER/nf-gwas-work. If blank, no subdir will be created. [Default: Blank]')
    parser.add_argument("--email4slurm", type=str, default="soumick.chatterjee@fht.org", help="Email ID to be used for SLURM notifications.")
    parser.add_argument("--nodes2exclude", type=str, default="", help="Coma-separated list of nodes to exclude.")

    parser.add_argument('-lg', '--launch_gwas', action=argparse.BooleanOptionalAction, default=True, help='Whether to launch the GWAS or not. [Default: True] If only to process the embeddings, we can set it to False.')
    
    #The following parameters will only be used if the phenotype file is not provided
    parser.add_argument('-nl', '--nLatents', type=int, default=128, help='Number of latent factors. [Default: 128]')
    parser.add_argument('-sl', "--selectiveLatents", type=str, default=None, help="Comma separated list of latent factors to be used for the GWAS. [Default: None]")
    parser.add_argument('-i', '--in_root_path', type=str, default="/project/ukbblatent/Out/Results", help='Path to the model folder (containing a folder with the trainID).')
    parser.add_argument('-is', '--in_train_subfold', type=str, default="", help='Subfolder inside the model (trainID) folder containing the embeddings. [Default: "" (Output_fullDS)]')
    parser.add_argument('-cd', '--clinical_data_path', type=str, default="/project/ukbblatent/clinicaldata", help='Path where there are the clinical data TSVs (to be used for filters).')

    parser.add_argument("--subIDs", help="Coma-seperated list of subject IDs. Blank for all.", default="")
    parser.add_argument("--filter_genethnicity", action=argparse.BooleanOptionalAction, help="Whether to filter the subjects based on genetic thenicity [True implies only white british will be included]", default=True)

    parser.add_argument("--data_tags", help="Coma-seperated list of data tags. Blank for all.", default="primary_LAX_4Ch_transverse_0")
    parser.add_argument("--flatten_embs", action=argparse.BooleanOptionalAction, help="Whether to flatten the data before storing or not.", default=False)
    parser.add_argument("--quantile_transform", action=argparse.BooleanOptionalAction, help="Whether to apply quantile transformer before preparing the final dataframe.", default=True)
    

    return parser

def filter_subs(args):
    filtered_subs = []
    flag = False
    if args['subIDs'] != "":
        filtered_subs.append(list(map(int, args['subIDs'].split(","))))
        flag = True
    if args['filter_genethnicity']:
        genetic_ethnicity = pd.read_table(f"{args['clinical_data_path']}/genetic_ethnicity.tsv").set_index('f.eid')
        filtered_subs.append(list(genetic_ethnicity.dropna().index))
        flag = True
    if len(filtered_subs) == 1:
        args['subIDs'] = ','.join(map(str, filtered_subs[0]))
    elif len(filtered_subs) > 1:
        args['subIDs'] = ','.join(map(str, set.intersection(*map(set, filtered_subs))))
    if flag and args['subIDs'] == "":
        sys.exit("No subjects left after filtering. Exiting...")


def prep_args_processEmbs(args):
    args_processEmbs = DotDict()
    args_processEmbs.in_path = f"{args['in_root_path']}/{args['trainID']}/{args['in_train_subfold']}/emb.h5"
    args_processEmbs.out_path = args['out_path']
    args_processEmbs.subIDs = args['subIDs']
    args_processEmbs.data_tags = args['data_tags']
    args_processEmbs.flatten_data = args['flatten_embs']
    args_processEmbs.save_csv = False
    args_processEmbs.save_tag = ""
    return args_processEmbs

def prep_args_conf(args, args4conf):
    if args['selectiveLatents'] is not None:
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
        "phenotypes_columns": latents
    }

    return args4conf

def create_conf(args, args4conf, tag=""):
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

    conf.save(f"{args['out_path']}/run_gwas{tag}.conf")
    print("Conf file created!")

def create_sbatch(args, phenotypes="", tag=""):
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

    if args['projectName'] == "":
        assert args['trainID'] != "", "Either trainID or projectName must be provided!"
        args['projectName'] = args['trainID']

    if args['filter_genethnicity'] and "WBRIT" not in args['projectName']:
        args['projectName'] = f"WBRIT_{args['projectName']}"

    if args['quantile_transform'] and "Qntl" not in args['projectName']:
        args['projectName'] = f"Qntl_{args['projectName']}"

    if bool(args['path_phenotypes']):
        if args['out_path'] == "":
            args['out_path'] = f"{os.path.dirname(args['path_phenotypes'])}/GWAS"
            if args['out_subdir_tag'] != "":
                args['out_path'] = f"{args['out_path']}_{args['out_subdir_tag']}"
    else:
        if args['in_train_subfold'] == "":
            args['in_train_subfold'] = "Output_fullDS"

        if args['out_path'] == "":
            args['out_path'] = f"{args['in_root_path']}/{args['trainID']}/GWAS"
            if args['out_subdir_tag'] != "":
                args['out_path'] = f"{args['out_path']}_{args['out_subdir_tag']}"
            else:
                args['out_path'] = f"{args['out_path']}_fullDS"
        else:
            args['out_path'] = f"{args['out_path']}/{args['projectName']}"
    os.makedirs(args['out_path'], exist_ok=True)

    if args['pipeline_root'] != "" and args['pipeline_root'][-1] != "/":
        args['pipeline_root'] = args['pipeline_root']+"/"

    if args['custom_work_dir']:
        args4conf['workDir'] = f"/scratch/{getpass.getuser()}/nf-gwas-work/{args['projectName']}"
        os.makedirs(args4conf['workDir'], exist_ok=True)

    #filter subjects [if needed]
    filter_subs(args)

    if args['path_phenotypes']:
        print("Phenotype file supplied! Skipping the processing of the emb.h5, rather processing the tsv file directly...")
        if "_raw" in args['path_phenotypes']:
            print("Raw phenotype file detected! Processing...")
            df = pd.read_table(args['path_phenotypes'], index_col="FID")
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
        # Processing the embeddings and saving them as an npy file
        embsH5_to_npy(prep_args_processEmbs(args))

        # Finalise the latent DFs (NPYs) by processing the final steps and then save them as a TSV file
        if os.path.isfile(f"{args['out_path']}/emb.npy"):
            finalise_latentDFs(f"{args['out_path']}/emb.npy", f"{args['out_path']}/processed.tsv", quantile_transform=args['quantile_transform'])
            args['path_phenotypes'] = f"{args['out_path']}/processed.tsv"
            gwas_ready = True
        else:
            print("No emb.npy found, also path_phenotypes has not been supplied! GWAS engager will be terminated if emb_mag.npy also not found...")

    if gwas_ready:
        #create conf file for GWAS pipeline
        args4conf = prep_args_conf(args, args4conf)
        create_conf(args, args4conf)
        
        phenotypes = args4conf['params']['phenotypes_columns']

        #create SLURM script and launch
        create_sbatch(args, phenotypes=phenotypes)
        if args['launch_gwas']:
            os.system(f"sbatch {args['out_path']}/run_gwas.sh")
            print("SLURM script sbatch-ed! Terminating GWAS engager...")
            
    elif os.path.isfile(f"{args['out_path']}/emb_mag.npy"):
        print("Complex embeddings NPY file detected! Processing...")
        for k in ["real", "imag", "mag", "phase"]:
            finalise_latentDFs(f"{args['out_path']}/emb_{k}.npy", f"{args['out_path']}/processed_{k}.tsv", quantile_transform=args['quantile_transform'])
            
            args_tmp, args4conf_tmp = deepcopy(args), deepcopy(args4conf)
            args_tmp["projectName"] += "_" + k
            args4conf_tmp['workDir'] += "_" + k
            args4conf_tmp = prep_args_conf(args_tmp, args4conf_tmp)
            if args["path_phenotypes"]:
                print("path_phenotypes cannot be used with complex embeddings! Terminating GWAS engager...")
                sys.exit(1)
            else:
                args4conf_tmp['params']['phenotypes_filename'] = args4conf_tmp['params']['phenotypes_filename'].replace("processed.tsv", f"processed_{k}.tsv")

            create_conf(args_tmp, args4conf_tmp, tag="_"+k)
            
            phenotypes = args4conf_tmp['params']['phenotypes_columns']

            #create SLURM script and launch
            create_sbatch(args_tmp, phenotypes=phenotypes, tag="_"+k)
            print("As the embeddings are complex, the GWAS pipeline needs to run for each of the real, imaginary, magnitude and phase embeddings. Hence, only prepared the SLURM script, but not launching them.")
    else:
        print("Neither emb.npy nor emb_mag.npy found, while path_phenotypes has also not been supplied! Terminating GWAS engager...")
        sys.exit(1)