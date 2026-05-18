#!/bin/bash

#creation of argument helper
programmename=$0
function usage {
    echo ""
    echo "Check analyses/binary_dis_MLAnalyses_disjobs.sh script for the arguments!"
    echo ""
}


#function to handle the death of the script!
function die {
    printf "Script failed: %s\n\n" "$1"
    exit 1
}

args=("$@")
while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

##############################################################################
# Defination of the default paramters
##############################################################################

job_name="${job_name:-259mag}"
sbatcher="${sbatcher:-/home/soumick.chatterjee/cpurun.sh}"
conda="${conda:-/project/ukbblatent/envs/torchHTBeta2V2UKB}"
tricorder_root="${tricorder_root:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder}"
programme="${programme:-analyses/latents/analysers/bin_dis_mlanalyses_summeriser.py}"

out_root="${out_root:-/project/ukbblatent/soumick/ML_analyses/Atlas/MultiOrganV3_Initial}"

trainID="${trainID:-F20259v3_Pancreas_gen2_initrunfold0_prec16-mixed_DiffAE}"
run_tag="${run_tag:-Mag}"

##############################################################################
# launch the array job for the ML analysing binary disease cohorts
##############################################################################

# array_jobid=$(sbatch -J ${job_name}_arr analyses/binary_dis_MLAnalyses_disjobs.sh "${args[@]}" | awk '{print $4}')
# echo "Array job for binary disease analyses submitted with JobID=${array_jobid}"

##############################################################################
# launch the job for the ML analysing IDPs and other phenos
##############################################################################

pheno_jobid=$(sbatch -J ${job_name}_pheno analyses/pheno_MLAnalyses.sh "${args[@]}" | awk '{print $4}')
echo "Job for pheno analyses submitted with JobID=${pheno_jobid}"

##############################################################################
# Process some params and launch the summeriser
##############################################################################

codes=("Phecode" "PhecodeCh" "PhecodeLvl1")
modes=("diag" "diag10y" "diag5y" "prog5y" "prog")

codes=$(IFS=,; echo "${codes[*]}")
modes=$(IFS=,; echo "${modes[*]}")

ds="${trainID%%_*}" # Extract everything before the first '_' from the trainID
model="${trainID##*_}"
organ="${trainID#*_}"  # Remove everything before the first underscore
organ="${organ%%_*}"

if [[ $organ == Brain* ]]; then
    organ="Brain"
fi

model_tag="${model}_${ds}_${organ}_${run_tag}"
echo "Model tag (inside main launcher): ${model_tag}"

# sbatch --dependency=afterok:${array_jobid} -J ${job_name}_consolDis ${sbatcher} --root ${tricorder_root} --programme ${programme} --conda ${conda} --args "--out_root ${out_root} --codes ${codes} --modes ${modes} --model_tag ${model_tag} --organ ${organ}"
# sbatch -J ${job_name}_consolDis ${sbatcher} --root ${tricorder_root} --programme ${programme} --conda ${conda} --args "--out_root ${out_root} --codes ${codes} --modes ${modes} --model_tag ${model_tag} --organ ${organ}"