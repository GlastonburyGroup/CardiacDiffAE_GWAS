#!/bin/bash
#SBATCH --job-name toploci
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --chdir=/home/soumick.chatterjee/SLURM/toploci
#SBATCH --output toploci_%x_%j.log
#SBATCH --partition cpuq
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=5000Mb
#SBATCH --time 10:00:00

# Load necessary modules
module use /ssu/gassu/software/modulefiles
module load plink/1.9_20210606

###function definations

#creation of argument helper
programmename=$0
function usage {
    echo ""
    echo "Logs a SLURM job with a GPU Node."
    echo ""
    echo "usage: $programmename --phenos string --minINFO number --saveFiltered logical --indir string --tmpdirpth string --outdir string --bfiledir string"
    echo ""
    echo "  --phenos        string         Coma-seperated list of phenotypes [Default: Z0 to Z127]"
	echo "  --minINFO       number         Min INFO score for imputed genotypes [Default: 0]"
	echo "  --saveFiltered  logical        Whether to save the filtered REGENIE output [Default: false]"
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
minINFO=${minINFO:-0.0}
saveFiltered=${saveFiltered:-false}
indir="${indir:-/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results}"
pth_sumstats="${pth_sumstats:-${indir}/allmerged_sig_nonrare.tsv}"
tmpdirpth="${tmpdirpth:-${indir}/tmp_toploci}"
outdir="${outdir:-${indir}/allmerged_sig}"
bfiledir="${bfiledir:-/processing_data/shared_datasets/ukbiobank/GWAS_pipeline/LD_panel/}"


PHENOTYPE="$(basename "$pth_sumstats" | cut -d. -f1)"
filteredFile="${indir}/${PHENOTYPE}.filtered.INFO.${minINFO}.gwas.regenie"
PthTMP="${tmpdirpth}_${PHENOTYPE}/"
PthOUT="${outdir}/"

#check if minINFO is 0.0
if [ "$minINFO" == "0.0" ]; then
    toplociFile="${PthOUT}/${PHENOTYPE}.toploci.tsv"
else
    toplociFile="${PthOUT}/${PHENOTYPE}.INFO.${minINFO}.toploci.tsv"
fi

#Exit if the toploci for this INFO score have already been extracted
if [ -f "$PthOUT/$toplociFile" ]; then
    echo "$PthOUT/$toplociFile exists. So, skipping processing phenotype: $PHENOTYPE"
    exit 0
fi

# Create directories if they don't exist
mkdir -p "${PthTMP}"
mkdir -p "${PthOUT}"

# Change to temp directory
cd "${PthTMP}"

# Filter the GWAS by INFO score adding the P value column
echo "INFO score filtering + PVAL calculation! Current time: $(date '+%Y-%m-%d %H:%M:%S')"
zcat "${pth_sumstats}" | awk -v minINFO=$minINFO '{OFS="\t"}; NR == 1 {print $0, "PVAL"}; NR > 1 {if ($7 >= minINFO) print $0, 10^(-$13)}' > filteredFile
echo "zcat & filtering done! Current time: $(date '+%Y-%m-%d %H:%M:%S')"

echo "Starting clumping! Current time: $(date '+%Y-%m-%d %H:%M:%S')"
# Loop over each chromosome
for chrom in {1..22}; do
    # Formatted variables
    bfilePrefix="chr${chrom}_mac100_30000_random_unrelated_white_british"
    bfile="${bfiledir}${bfilePrefix}"

    # Generate the clump files
    echo "chromosome ${chrom} working! Current time: $(date '+%Y-%m-%d %H:%M:%S')"
    plink --memory 4096 --bfile "${bfile}" --chr "${chrom}" --clump filteredFile --clump-p1 5E-8 --clump-p2 0.0001 --clump-kb 250 --clump-r2 0.5 --clump-snp-field ID --clump-field PVAL --out "${chrom}"
    echo "plink done! Current time: $(date '+%Y-%m-%d %H:%M:%S')"

    touch "${chrom}.clumped"
    # touch "${chrom}.clumped.ranges"
done

# Merge results
echo -e "CHR\tF\tSNP\tBP\tP\tTOTAL\tNSIG\tS05\tS01\tS001\tS0001\tSP2" > "$toplociFile"
for f in *.clumped
do
    tail -n+2 "${f}" | tr -s " " "\t" | sed 's/^\t//g' >> toploci.tsv
done
sed '/^$/d' toploci.tsv | sort -k5,5g >> "$toplociFile"
mv "$toplociFile" "${PthOUT}"
echo "$toplociFile done! Current time: $(date '+%Y-%m-%d %H:%M:%S')"

# Save the filtered results (removing the P VALUE has been attached as column #15)
if [ "$minINFO" != "0.0" ] && ($saveFiltered); then
    echo "Saving filtered GWAS results! Current time: $(date '+%Y-%m-%d %H:%M:%S')"
    mv filteredFile $filteredFile
    gzip $filteredFile
    echo "Saving done! Current time: $(date '+%Y-%m-%d %H:%M:%S')"
fi


# echo -e "CHR\tSNP\tP\tN\tPOS\tKB\tRANGES" > "${PHENOTYPE}.toploci.annot.tsv"
# for f in *.clumped.ranges
# do
#     tail -n+2 "${f}" | tr -s " " "\t" | sed 's/^\t//g' >> toploci.annot.tsv
# done
# sed '/^$/d' toploci.annot.tsv | sort -k3,3g >> "${PHENOTYPE}.toploci.annot.tsv"
# mv "${PHENOTYPE}.toploci.annot.tsv" "${PthOUT}"
# echo "toploci.annot.tsv done! Current time: $(date '+%Y-%m-%d %H:%M:%S')"

# Clean up the temp directory
cd ..
rm -r "${PthTMP}"
