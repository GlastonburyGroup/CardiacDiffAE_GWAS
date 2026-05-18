#!/bin/bash
#SBATCH --job-name=getrelPheno
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
  --bgen PATH           Path to input BGEN file
  --sample PATH         Path to input sample file
  --keep PATH           Path to file with IDs to keep
  --king_cutoff VALUE   King_cutoff value for relatedness filtering
  --out PATH            Output path prefix
  
Optional parameters:
  --threads N           Number of threads (default: 64)
  --help                Show this help message

Example:
  $0 --bgen /path/to/input.bgen --sample /path/to/input.sample --keep /path/to/ids.txt --king_cutoff 0.0625 --out /path/to/output
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
: ${bgen:="/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/lw_gw_indep/gwcond_plus_plink_maf1p_geno10p_caucasian_prune_250_5_r0p5_ukbb_autosomes_mac100_info0p4.bgen"}
: ${keep:="/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/subIDs_nonDisc_caucas.txt"}
: ${king_cutoff:=0.0625}
: ${threads:=64}

#check if output path is provided, else derive from extract path
if [ -z "$out" ]; then
    bgen_basename=$(basename "${bgen%.*}")
    king_cutoff_formatted="${king_cutoff/./p}"
    out="$(dirname "$bgen")/king_cutoff_${king_cutoff_formatted}_nonDisc_${bgen_basename}"
fi

#check if sample path is provided, else use the bgen path by replacing .bgen with .sample
if [ -z "$sample" ]; then
    sample="${bgen%.bgen}.sample"
fi

# Check required parameters
if [ -z "$bgen" ] || [ -z "$sample" ] || [ -z "$keep" ] || [ -z "$king_cutoff" ] || [ -z "$out" ]; then
    echo "Error: Missing required parameters!"
    echo ""
    usage
    exit 1
fi

env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG

module load mpi/2021.2.0
module load plink/2.00_20211217

#plink2 --bgen your_genotype_file.bgen --sample your_sample_file.sample --indep-pairwise <window size> <step> <r2 threshold>

plink2 \
    --bgen "$bgen" ref-first \
    --sample "$sample" \
    --keep "$keep" \
    --king-cutoff $king_cutoff \
    --out "$out" \
    --threads $threads