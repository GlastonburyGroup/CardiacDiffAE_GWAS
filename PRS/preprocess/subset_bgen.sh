#!/bin/bash
#SBATCH --job-name=subsetBGEN
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=30-00:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=5    # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=plink2_%x_%j.log
#SBATCH --mem=300G # RAM per CPU

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required parameters:
  --bgen PATH                   Path to input BGEN file (default: /scratch/edoardo.giacopuzzi/UKBB/step2_dataset/step2_dataset_autosomes.mac100.bgen)
  --sample PATH                 Path to input sample file (default: /scratch/edoardo.giacopuzzi/UKBB/step2_dataset/step2_dataset_autosomes.mac100.sample)
  --keep PATH                   Path to file with IDs to keep (default: /group/glastonbury/soumick/GWAS/ukbb_genotypes/subIDs_Caucasian.txt)
  --pruned_SNPs PATH            Path to file with pruned SNPs (default: /group/glastonbury/soumick/PRS/inputs/common/prunedUKBB/plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.txt)
  --gwcond_SNPs_postCOJO PATH   Path to file with GWAS significant SNPs after COJO (default: /group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/lw_gw_indep/gwcond_SNPs_postCOJO.txt)
  --extract PATH                Output path to file with SNPs to extract (default: derived from gwcond_SNPs_postCOJO's directory and pruned_SNPs filename, in the format: <directory_of_gwcond>/gwcond_plus_<basename_of_prunedSNPs>)
  --out PATH                    Output path prefix (default: extract path without extension)
  
Optional parameters:
  --threads N           Number of threads (default: 64)
  --bgenix PATH         Path to bgenix executable (default: /ssu/gassu/software/bgen/1.1.7/bgenix)
  --help                Show this help message

Example:
  $0 --bgen /path/to/input.bgen --sample /path/to/input.sample --keep /path/to/ids.txt --extract /path/to/snps.txt --out /path/to/output
EOF
}

exec 2>&1      # send errors into stdout stream

# Read the different keyworded commandline arguments
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

# Set default values after reading arguments
: ${bgen:="/scratch/edoardo.giacopuzzi/UKBB/step2_dataset/step2_dataset_autosomes.mac100.bgen"}
: ${keep:="/group/glastonbury/soumick/GWAS/ukbb_genotypes/subIDs_Caucasian.txt"}
: ${pruned_SNPs:="/group/glastonbury/soumick/PRS/inputs/common/prunedUKBB/plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.txt"}
: ${gwcond_SNPs_postCOJO:="/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/lw_gw_indep/gwcond_SNPs_postCOJO.txt"}
: ${extract:="$(dirname "$gwcond_SNPs_postCOJO")/gwcond_plus_$(basename "$pruned_SNPs")"}
: ${threads:=64}
: ${bgenix:=/ssu/gassu/software/bgen/1.1.7/bgenix}

#check if output path is provided, else derive from extract path
if [ -z "$out" ]; then
    out="${extract%.*}"
fi

#check if sample path is provided, else use the bgen path by replacing .bgen with .sample
if [ -z "$sample" ]; then
    sample="${bgen%.bgen}.sample"
fi

# Check required parameters
if [ -z "$bgen" ] || [ -z "$sample" ] || [ -z "$keep" ] || [ -z "$extract" ] || [ -z "$out" ]; then
    echo "Error: Missing required parameters!"
    echo ""
    usage
    exit 1
fi

env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG

# Create union of pruned_SNPs and gwcond_SNPs_postCOJO and store in extract
echo "Creating union of SNP lists for extraction from pruned_SNPs and gwcond_SNPs_postCOJO and storing in $extract"
cat "$pruned_SNPs" "$gwcond_SNPs_postCOJO" | sort -u > "$extract"

module load mpi/2021.2.0
module load plink/2.00_20211217

plink2 \
    --bgen "$bgen" ref-first \
    --sample "$sample" \
    --keep "$keep" \
    --extract "$extract" \
    --export bgen-1.2 bits=8 \
    --out "$out" --threads $threads

echo "BGEN subsetted, now indexing..."
"$bgenix" -g "${out}.bgen" -index