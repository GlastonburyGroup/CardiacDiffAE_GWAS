#!/bin/bash
#SBATCH --job-name=evalmPRSV2ppr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=cpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=10:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=1    # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=evalprs_%x_%j.log
#SBATCH --mem-per-cpu=18000Mb # RAM per CPU

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

# #read config file for the supplied arguments
# if [ -n "$config_file" ]; then
#   source "$config_file"
# else
#   echo "Error: Configuration file not found."
#   exit 1
# fi

#set the default values for the commandline arguments

#general arguments
root="${root:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder/}"
programme="${programme:-PRS/eval_multiPRS_V2_wCovar.py}"
conda="${conda:-/scratch/soumick.chatterjee/conda_envs/analyseNvis}"

#arguments for the programme
output_root="${output_root:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/newcovsets_V0v2/4paper_caucasian_king0p0625_grouped}"
out_tag="${out_tag:-panCohortV2_auto_lw_gw_10kIT_kingB4ldpred2}"

disease_root="${disease_root:-/project/ukbblatent/clinicaldata/binary_disease_cohorts/F20208v3_nonDiscov/caucasian_king0p0625_grouped/newcovsets/V0v2}"

prs_res_root="${prs_res_root:-/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/}"
rds_pres_prefix="${rds_pres_prefix:-run_ext_basic_king0p0625_lw_gw_indep_FiltMAF_}"
rds_pres_suffix="${rds_pres_suffix:-.fullDS.auto.mod.LDPred2.rds}" #for inf model: .fullDS.inf.mod.LDPred2.rds , for auto model: .fullDS.auto.mod.LDPred2.rds , for grid model: .fullDS.grid.mod.LDPred2.rds
rds_tag_prs="${rds_tag_prs:-auto.mod}" #for inf model: inf.mod , for auto model: auto.mod , for grid model: grid.mod
tag_data="${tag_data:-resNdata.basic}"
tag_prs="${tag_prs:-pred_auto}" #for inf model: pred_inf , for auto model: pred_auto , for grid model: pred_grid

ext_covar="${ext_covar:-/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/covars/nonDisc_caucasian_king0p0625_V0.tsv}"
covar_cont_cols="${covar_cont_cols:-Age,BMI}"
covar_cat_cols="${covar_cat_cols:-Sex,CAT_Smoking}"
adjust_prs_for_covars="${adjust_prs_for_covars:-1}"

raw_disease_path="${raw_disease_path:-/project/ukbblatent/clinicaldata/merge_SR_HI_GP_v4_allUKB_&_HEALTHY.csv}"
raw_baseline_path="${raw_baseline_path:-/project/ukbblatent/clinicaldata/v1.1.0_seventh_basket/baseline_MD_27_10_2023_13_10_05.tsv}"
raw_centre_info_path="${raw_centre_info_path:-/project/ukbblatent/clinicaldata/v1.1.0_seventh_basket/assessmentCentre_82779_MD_13_06_2024_12_18_59.tsv}"

drop_dis="${drop_dis:-extended arrythmias atrial_ventricular,extended miscellaneous,miscellaneous,cardiac arrest,pericardial problem}"

is_pancohort="${is_pancohort:-1}"
save_plots="${save_plots:-1}"
sex_stratified="${sex_stratified:-1}"

obtain_summary="${obtain_summary:-0}"
obtain_pairwise_improvements="${obtain_pairwise_improvements:-0}"

plot_box_AUC="${plot_box_AUC:-0}"
plot_box_F1="${plot_box_F1:-0}"
plot_box_logOR="${plot_box_logOR:-0}"
plot_prevalence_prob_diseasecohort="${plot_prevalence_prob_diseasecohort:-0}"
plot_prevalence_prs_pancohort="${plot_prevalence_prs_pancohort:-1}"
plot_prevalence_prob_pancohort="${plot_prevalence_prob_pancohort:-0}"

plot_cum_disease_burden="${plot_cum_disease_burden:-1}"
plot_cum_hazard="${plot_cum_hazard:-1}"
plot_KM_survival="${plot_KM_survival:-1}"
plot_cox_ph="${plot_cox_ph:-1}"

dis_plots_mod="${dis_plots_mod:-GLM,singlePRS,singlePRSCovar}"
dis_plots_rawPRS="${dis_plots_rawPRS:-1}"
dis_plots_cutoff_date="${dis_plots_cutoff_date:-2023-10-31}"
dis_plots_upto_Nyears="${dis_plots_upto_Nyears:-10}"

quick_run="${quick_run:-0}"
quick_run_sex="${quick_run_sex:-}"
quick_run_diseases="${quick_run_diseases:-}"

export_programme="${export_programme:-PRS/4paper_export_plot_tsvs.py}"
run_export="${run_export:-1}"
output_root_prognosis="${output_root_prognosis:-}"
export_out_dir="${export_out_dir:-$output_root/$out_tag/paper_tsvs}"


###
###Start of the actual script, after reading all the arguments

#Setup conda
source /home/${USER}/.bashrc
conda activate $conda

cd $root
# $conda/bin/python $programme \
#     --output_root "$output_root/$out_tag" \
#     --disease_root "$disease_root" \
#     --rds_pres_prefix "$prs_res_root/$rds_pres_prefix" \
#     --rds_pres_suffix "$rds_pres_suffix" \
#     --rds_tag_prs "$rds_tag_prs" \
#     --tag_data "$tag_data" \
#     --tag_prs "$tag_prs" \
#     --ext_covar "$ext_covar" \
#     --covar_cont_cols "$covar_cont_cols" \
#     --covar_cat_cols "$covar_cat_cols" \
#     $( [[ "$adjust_prs_for_covars" -eq 1 ]] && echo '--adjust_prs_for_covars' || echo '--no-adjust_prs_for_covars' ) \
#     --raw_disease_path "$raw_disease_path" \
#     --raw_baseline_path "$raw_baseline_path" \
#     --raw_centre_info_path "$raw_centre_info_path" \
#     --drop_dis "$drop_dis" \
#     $( [[ "$is_pancohort" -eq 1 ]] && echo '--is_pancohort' || echo '--no-is_pancohort' ) \
#     $( [[ "$save_plots" -eq 1 ]] && echo '--save_plots' || echo '--no-save_plots' ) \
#     $( [[ "$sex_stratified" -eq 1 ]] && echo '--sex_stratified' || echo '--no-sex_stratified' ) \
#     $( [[ "$obtain_summary" -eq 1 ]] && echo '--obtain_summary' || echo '--no-obtain_summary' ) \
#     $( [[ "$obtain_pairwise_improvements" -eq 1 ]] && echo '--obtain_pairwise_improvements' || echo '--no-obtain_pairwise_improvements' ) \
#     $( [[ "$plot_box_AUC" -eq 1 ]] && echo '--plot_box_AUC' || echo '--no-plot_box_AUC' ) \
#     $( [[ "$plot_box_F1" -eq 1 ]] && echo '--plot_box_F1' || echo '--no-plot_box_F1' ) \
#     $( [[ "$plot_box_logOR" -eq 1 ]] && echo '--plot_box_logOR' || echo '--no-plot_box_logOR' ) \
#     $( [[ "$plot_prevalence_prob_diseasecohort" -eq 1 ]] && echo '--plot_prevalence_prob_diseasecohort' || echo '--no-plot_prevalence_prob_diseasecohort' ) \
#     $( [[ "$plot_prevalence_prs_pancohort" -eq 1 ]] && echo '--plot_prevalence_prs_pancohort' || echo '--no-plot_prevalence_prs_pancohort' ) \
#     $( [[ "$plot_prevalence_prob_pancohort" -eq 1 ]] && echo '--plot_prevalence_prob_pancohort' || echo '--no-plot_prevalence_prob_pancohort' ) \
#     $( [[ "$plot_cum_disease_burden" -eq 1 ]] && echo '--plot_cum_disease_burden' || echo '--no-plot_cum_disease_burden' ) \
#     $( [[ "$plot_cum_hazard" -eq 1 ]] && echo '--plot_cum_hazard' || echo '--no-plot_cum_hazard' ) \
#     $( [[ "$plot_KM_survival" -eq 1 ]] && echo '--plot_KM_survival' || echo '--no-plot_KM_survival' ) \
#     $( [[ "$plot_cox_ph" -eq 1 ]] && echo '--plot_cox_ph' || echo '--no-plot_cox_ph' ) \
#     --dis_plots_mod "$dis_plots_mod" \
#     $( [[ "$dis_plots_rawPRS" -eq 1 ]] && echo '--dis_plots_rawPRS' || echo '--no-dis_plots_rawPRS' ) \
#     --dis_plots_cutoff_date "$dis_plots_cutoff_date" \
#     --dis_plots_upto_Nyears "$dis_plots_upto_Nyears" \
#     $( [[ "$quick_run" -eq 1 ]] && echo '--quick_run' || echo '--no-quick_run' ) \
#     $( [[ -n "$quick_run_sex" ]] && echo "--quick_run_sex '$quick_run_sex'" || echo "" ) \
#     $( [[ -n "$quick_run_diseases" ]] && echo "--quick_run_diseases '$quick_run_diseases'" || echo "" )

if [[ "$run_export" -eq 1 ]]; then
    prognosis_root="${output_root_prognosis:-$output_root/$out_tag}"
    $conda/bin/python $export_programme \
        --output_root_diagnosis "$output_root/$out_tag" \
        --output_root_prognosis "$prognosis_root" \
        $( [[ "$adjust_prs_for_covars" -eq 1 ]] && echo '--covar_adjusted' || echo '--no-covar_adjusted' ) \
        --out_dir "$export_out_dir"
fi