#!/bin/bash
#SBATCH --job-name toploci
#SBATCH --mail-type=ALL
#SBATCH --output toploci_%x_%j.log
#SBATCH --partition cpuq
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=5000Mb
#SBATCH --time 10:00:00
#SBATCH --array=0-127

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
    echo "usage: $programmename --phenos string --minINFO number --saveFiltered logical --indir string --tmpdirpth string --outdir string --bfiledir string"
    echo ""
    echo "  --phenos        string         Coma-seperated list of phenotypes [Default: Z0 to Z127]"
	echo "  --minINFO       number         Min INFO score for imputed genotypes [Default: 0]"
	echo "  --saveFiltered  logical        Whether to save the filtered REGENIE output [Default: false]"
	echo "  --minINFO       number         Min INFO score for imputed genotypes [Default: 0]"
	echo "  --saveFiltered  logical        Whether to save the filtered REGENIE output [Default: false]"
    echo "  --indir         string         Path to the input directory containing the GWAS results [Default: Nothing/Blank]"
    echo "  --tmpdirpth     string         Path to store the temporary files (_NameOfPhenotype will be appended) [Default: indir/tmp_toploci]"
    echo "  --outdir        string         Path to store the output files [Default: indir/toploci]"
    echo "  --bfiledir      string         Path to the directory containing the LD panels [Default: /processing_data/shared_datasets/ukbiobank/GWAS_pipeline/LD_panel/]"
    echo ""
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
phenos="${phenos:-Z0,Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9,Z10,Z11,Z12,Z13,Z14,Z15,Z16,Z17,Z18,Z19,Z20,Z21,Z22,Z23,Z24,Z25,Z26,Z27,Z28,Z29,Z30,Z31,Z32,Z33,Z34,Z35,Z36,Z37,Z38,Z39,Z40,Z41,Z42,Z43,Z44,Z45,Z46,Z47,Z48,Z49,Z50,Z51,Z52,Z53,Z54,Z55,Z56,Z57,Z58,Z59,Z60,Z61,Z62,Z63,Z64,Z65,Z66,Z67,Z68,Z69,Z70,Z71,Z72,Z73,Z74,Z75,Z76,Z77,Z78,Z79,Z80,Z81,Z82,Z83,Z84,Z85,Z86,Z87,Z88,Z89,Z90,Z91,Z92,Z93,Z94,Z95,Z96,Z97,Z98,Z99,Z100,Z101,Z102,Z103,Z104,Z105,Z106,Z107,Z108,Z109,Z110,Z111,Z112,Z113,Z114,Z115,Z116,Z117,Z118,Z119,Z120,Z121,Z122,Z123,Z124,Z125,Z126,Z127}"
minINFO=${minINFO:-0.0}
saveFiltered=${saveFiltered:-false}
indir="${indir:-}"
tmpdirpth="${tmpdirpth:-${indir}/tmp_toploci}"
outdir="${outdir:-${indir}/toploci}"
bfiledir="${bfiledir:-/processing_data/shared_datasets/ukbiobank/GWAS_pipeline/LD_panel/}"

# Convert the comma-separated string of phenotypes to an array
IFS=',' read -r -a PHENOTYPES <<< "$phenos"

# Get the phenotype for this array task
PHENOTYPE=${PHENOTYPES[$SLURM_ARRAY_TASK_ID]}

PhenoGZ="${indir}/${PHENOTYPE}.gwas.regenie.gz"
filteredFile="${indir}/${PHENOTYPE}.filtered.INFO.${minINFO}.gwas.regenie"
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

echo "Processing phenotype: $PHENOTYPE"

# Create directories if they don't exist
mkdir -p "${PthTMP}"
mkdir -p "${PthOUT}"

# Change to temp directory
cd "${PthTMP}"

# Filter the GWAS by INFO score adding the P value column
echo "INFO score filtering + PVAL calculation! Current time: $(date '+%Y-%m-%d %H:%M:%S')"
zcat "${PhenoGZ}" | awk -v minINFO=$minINFO '{OFS="\t"}; NR == 1 {print $0, "PVAL"}; NR > 1 {if ($7 >= minINFO) print $0, 10^(-$13)}' > filteredFile
echo "zcat & filtering done! Current time: $(date '+%Y-%m-%d %H:%M:%S')"

# Filter the GWAS by INFO score adding the P value column
echo "INFO score filtering + PVAL calculation! Current time: $(date '+%Y-%m-%d %H:%M:%S')"
zcat "${PhenoGZ}" | awk -v minINFO=$minINFO '{OFS="\t"}; NR == 1 {print $0, "PVAL"}; NR > 1 {if ($7 >= minINFO) print $0, 10^(-$13)}' > filteredFile
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
