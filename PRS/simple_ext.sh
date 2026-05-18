#!/bin/bash
#SBATCH --job-name=smplPRS
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=30-00:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=5    # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=prs_%x_%j.log
#SBATCH --mem-per-cpu=15000Mb # RAM per CPU

exec 2>&1      # send errors into stdout stream
env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG

module load mpi/2021.2.0

#Setup conda
source /home/${USER}/.bashrc
conda activate /ssu/gassu/conda_envs/bigsnprenv_2025

#Setup parameters 
threads="32"
summarycols="/group/glastonbury/soumick/MyCodes/GitLab/tricorder/PRS/davide_fede/sumcols_UKBB_regenie.json"
summary="/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/gwas/S1701_Z49.gwas.regenie.gz"
# summary="/group/glastonbury/soumick/MyCodes/GitLab/tricorder/PRS/davide_fede/S1701_Z49_map_hm3_ldpred2.gwas.regenie.gz"
#index="/group/soranzo/bigsnprtest_ukbiobk/30k.idx_bgen.tsv"
# reference="/processing_data/shared_datasets/ukbiobank/genotypes/LD_reference/ld_reference_bfiles/ukbb_all_30000_random_unrelated_white_british.bed"
# reference="/group/soranzo/bigsnprtest_ukbiobk/bed/ukbb_all_30000_random_unrelated_white_british.rds"
### assuming .bgen files are in a folder named "test"
### one can simply do ls test/*.bgen > bgen.fofn
### this will generate the proper .fofn file to use as input

filter_SNPs="/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/cond_plus_plink_maf1p_geno10p_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.txt"

phenotype="/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/validated_input/processed.pheno.validated.txt"
pheno_col="S1701_Z49"
covariates="/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/validated_input/cov_newset_chp_F20208_Long_axis_heart_images_DICOM_H5v3_NOnoise.cov.validated.txt"


# bgen="/group/glastonbury/soumick/GWAS/ukbb_genotypes/imputed/bgen.fofn"
# inpbgen="/group/glastonbury/soumick/GWAS/ukbb_imputed_ukb22828_v3_bgen.fofn"
# inpbgen="/processing_data/shared_datasets/ukbiobank/raw_data/genotypes/imputed/ukb22828_c5_b0_v3.bgen"
# inpbgen="/group/glastonbury/soumick/GWAS/ukbb_genotypes/imputed/ukb22828_c3_b0_v3.bgen"
# inpbgen="/processing_data/shared_datasets/ukbiobank/raw_data/genotypes/imputed/ukb22828_c3_b0_v3.bgen"
inpbgen="/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/conv_cond_plus_plink_maf1p_geno10p_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.bgen"
outdir="/group/glastonbury/soumick/PRS/davide_fede/initial_test_S1701_Z49_ldprune/cond_plus_plink_maf1p_geno10p_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4_bgen"

mkdir -p $outdir

cd /group/glastonbury/soumick/MyCodes/GitHub/prs_pipeline
Rscript scripts/simple_ext.r #\
    # --input $inpbgen \
    # --summary $summary --summarycols $summarycols --filter_SNPs $filter_SNPs \
    # --phenotype $phenotype --pheno_col $pheno_col --covariates $covariates \
    # --output $outdir --threads $threads 
#Rscript /group/glastonbury/soumick/MyCodes/GitHub/prs_pipeline/scripts/prs.r --input $inpbgen --threads $threads --output $outdir --summarycols $summarycols --summary $summary --reference $reference
#Closing statement
