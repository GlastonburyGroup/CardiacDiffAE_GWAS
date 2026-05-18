#!/bin/bash
#SBATCH --job-name=ldp2CardiIDP_nonDisc_king
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
##SBATCH --time=1-00:00:0      # walltime [Takes usually around 7-8 hours]
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=2    # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=prs_%x_%j.log
#SBATCH --mem=100G
#SBATCH --array=0-18

exec 2>&1      # send errors into stdout stream
env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG

module load mpi/2021.2.0

#creation of argument helper
programmename=$0
function usage {
    echo ""
    echo "Logs a SLURM job with a GPU Node."
    echo ""
    echo "usage: $programmename --root string --programme string --conda string --args \"string\" or \" \""
    echo ""
    echo "  --root    string           The program root, with the trailing slash (Default: /group/glastonbury/soumick/MyCodes/)"
    echo "                             [If no root is to be supplied, then put quotes with a space in between as its contents]"
    echo "  --programme string         The python file to run (Default: main.py)"
    echo "  --args    string           List of command line arguments, supplied as a single string within quotes (Default: Nothing/Blank)"
    echo "                             [Example: \"--modelID 2 --dataset UKB\". These will be supplied as arguments to the programme]"
    echo "  --conda   string           Name of the Conda env (Default: torchHTBeta2)"
    echo ""
}


#function to handle the death of the script!
function die {
    printf "Script failed: %s\n\n" "$1"
    exit 1
}

###
###process the arguments

#read the different keyworded commandline arguments
while [ $# -gt 0 ]; do
    if [[ $1 == "--help" ]]; then
        usage
        exit 0
    elif [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

#set the default values for the commandline arguments

#general arguments
root="${root:-/group/glastonbury/soumick/MyCodes/GitHub/prs_pipeline/}"
programme="${programme:-scripts/ldpred2_basic_ext_fullUKB.r}"
conda="${conda:-/ssu/gassu/conda_envs/bigsnprenv_2025}"

#arguments related to the pipeline, specific to LDPred2 (ldpred2_basic_ext_fullUKB.r)
input="${input:-/group/glastonbury/soumick/PRS/inputs/F20208_IDPs/lw_gw_indep/cardiac/gwcond_plus_plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.bgen}" 
pheno_SNPs="${pheno_SNPs:-/group/glastonbury/soumick/PRS/inputs/F20208_IDPs/lw_gw_indep/cardiac/lw_SNPs_gwcond_plus_plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4}"

# subs2include="${subs2include:-/group/glastonbury/soumick/PRS/inputs/F20208_IDPs/lw_gw_indep/cardiac/king_cutoff_0p0625_nonDisc_gwcond_plus_plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.king.cutoff.in.id}" #TODO
subs2include="${subs2include:-/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/2_king_cutoff_0p0625_nonDisc_cond_plus_plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.king.cutoff.in.id}" #using the same as the latent king ones, should not be that different (I think?)

output_prefix="${output_prefix:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/IDPs/cardiac/run_ext_basic_DiffAEking0p0625_lw_gw_indep_FiltMAF_}"

ext_sumstats_root="${ext_sumstats_root:-/group/glastonbury/GWAS/CarPheno/6thbasket_cardiac_phenos/Qntl_WBRIT_6thbasket_cardiac_phenos/Qntl_WBRIT_6thbasket_cardiac_phenos/results/gwas}"
ext_col_sumstats="${ext_col_sumstats:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder/PRS/sumcols_UKBB_regenie.json}"
filtMAF="${filtMAF:-1}"
threads="${threads:-5}"

##pheno_col will be parameterised in the script

###
###Start of the actual script, after reading all the arguments

#Setup conda
source /home/${USER}/.bashrc
conda activate $conda

#extract all the phenotype names
phenos=($(ls "${pheno_SNPs}"/*.txt | awk -F'[/.]' '{print $(NF-1)}'))
pheno_col=${phenos[$SLURM_ARRAY_TASK_ID]} 
echo "Processing Phenotype: $pheno_col"

#if "_AHA_" is in the phenotype name, then end the job here successfully
if [[ $pheno_col == *"_AHA_"* ]]; then
    echo "Phenotype $pheno_col is an AHA phenotype, skipping as we only want the global ones!"
    exit 0
fi

rsids2include="${pheno_SNPs}/${pheno_col}.txt"
ext_sumstats="$ext_sumstats_root/$pheno_col.gwas.regenie.gz"
output="${output_prefix}${pheno_col}"

cd $root

Rscript $programme \
        --input $input \
        --subs2include $subs2include \
        --rsids2include $rsids2include \
        --output $output \
        --ext_sumstats $ext_sumstats \
        --ext_col_sumstats $ext_col_sumstats \
        --filtMAF $filtMAF \
        --threads $threads 

#old code used the following defaults, but the covariates aren't used. So, removing them from the commandline arguments
# covariates="${covariates:-/group/glastonbury/soumick/PRS/inputs/cov_caucas_nonMRIandREP_cohort.tsv}" 
# cov_cols="${cov_cols:-Sex,Age,BSA}"
# cov_cat="${cov_cat:-Sex}"
# cov_PC="${cov_PC:-/project/ukbblatent/clinicaldata/v1.1.0_seventh_basket/genPC_82779_MD_01_03_2024_00_05_30.tsv}"
# cov_nPC="${cov_nPC:-20}"
# Rscript $programme \
#         --input $input \
#         --covariates $covariates \
#         --cov_cols $cov_cols \
#         --cov_cat $cov_cat \
#         --cov_PC $cov_PC \
#         --cov_nPC $cov_nPC \
#         --subs2include $subs2include \
#         --rsids2include $rsids2include \
#         --output $output \
#         --ext_sumstats $ext_sumstats \
#         --ext_col_sumstats $ext_col_sumstats \
#         --filtMAF $filtMAF \
#         --threads $threads 