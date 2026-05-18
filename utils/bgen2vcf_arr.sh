#!/bin/bash
#SBATCH --job-name bgen2vcf
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --chdir=/home/soumick.chatterjee/SLURM
#SBATCH --output bgen2vcf_%x_%j.log
#SBATCH --partition cpuq
#SBATCH --cpus-per-task 10
#SBATCH --mem-per-cpu=15000Mb
#SBATCH --time 24:00:00
#SBATCH --array=0-23

exec 2>&1      # send errors into stdout stream
env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG

module load mpi/2021.2.0
module load plink/2.00_20211217

###function definations

programmename=$0
function usage {
    echo ""
    echo "Logs a SLURM job with a GPU Node."
    echo ""
    echo "usage: $programmename --phenos string --indir string --tmpdirpth string --outdir string --bfiledir string"
    echo ""
    echo "  --phenos        string         Coma-seperated list of phenotypes [Default: Z0 to Z127]"
    echo "  --indir         string         Path to the input directory containing the GWAS results [Default: Nothing/Blank]"
    echo "  --tmpdirpth     string         Path to store the temporary files (_NameOfPhenotype will be appended) [Default: indir/tmp_toploci]"
    echo "  --outdir        string         Path to store the output files [Default: indir/toploci]"
    echo "  --bfiledir      string         Path to the directory containing the LD panels [Default: /processing_data/shared_datasets/ukbiobank/GWAS_pipeline/LD_panel/]"
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
bgenpattern="${bgenpattern:-ukb22828_cNCC1701_b0_v3.bgen}" #NCC1701 is the placeholder for the chromosome number
indir="${indir:-/processing_data/shared_datasets/ukbiobank/raw_data/genotypes/imputed}"
outdir="${outdir:-/group/glastonbury/soumick/GWAS/ukbb_genotypes/imputed_VCFs}"

# Get the chromosome number from the array index
CHROMOSOME=$((SLURM_ARRAY_TASK_ID+1))

if [ $CHROMOSOME -eq 23 ]; then
    CHROMOSOME="X"
elif [ $CHROMOSOME -eq 24 ]; then
    CHROMOSOME="XY"
fi

echo "Processing chromosome: $CHROMOSOME"

#create the input file name by replacing the placeholder with the chromosome number
bgenfile="${bgenpattern//NCC1701/$CHROMOSOME}"
echo "bgenfile: $bgenfile"

# create the sample file name by searching the indir for a file that starts with the name of the bgen file (except the extension) and sends with .sample
samplefile=$(ls $indir | grep "^${bgenfile%.bgen}.*\.sample$")
echo "samplefile: $samplefile"

outfile="${outdir}/${bgenfile%.bgen}.vcf.gz"
echo "outfile: $outfile"

# convert the bgen file to vcf
plink2 --bgen $indir/$bgenfile ref-first --sample $indir/$samplefile --export vcf bgz --out $outfile

echo "Completed processing chromosome: $CHROMOSOME"

