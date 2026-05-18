#!/usr/bin/env fish
# Example usage script for multi_prs_set_comparison_pipeline.py
# This demonstrates how to compare multiple PRS sets against each other

# Adjust these paths to your actual data locations
set PTH_COVARS "/group/glastonbury/soumick/PRS/inputs/F20208v3_DiffAE_select_latents_r80_discov_INF30/covars/nonDisc_caucasian_king0p0625_V0.tsv"
set PTH_DIS "/project/ukbblatent/clinicaldata/binary_disease_cohorts/F20208v3_nonDiscov/caucasian_king0p0625_grouped/newcovsets/V0v2/atherosclerotic.csv"
set PTH_OUT_ROOT "/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc/multi_set_comparisons"

# Example 1: Compare three pre-processed PRS sets
echo "Example 1: Comparing three PRS sets with Variational Inference (fast)"
echo "Note: First run will process RDS files and cache. Subsequent runs will be much faster!"
python multi_prs_set_comparison_pipeline.py \
  --pth_covars $PTH_COVARS \
  --pth_dis $PTH_DIS \
  --pth_out_root $PTH_OUT_ROOT \
  --pth_out_dir CVD_three_sets_comparison \
  --prs_set_name DiffAE_Custom \
  --prs_set_root "/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc" \
  --prs_set_name LDPred2_Alternative \
  --prs_set_root "/path/to/ldpred2_rds_root" \
  --prs_set_name PRS_CS_Alternative \
  --prs_set_root "/path/to/prs_cs_rds_root" \
  --reference_set DiffAE_Custom \
  --target_col BinCAT_Disease \
  --covariates Age BMI \
  --cat_covariates Sex CAT_Smoking \
  --orthogonalise \
  --use_variational_inference \
  --run_elastic_net \
  --run_pca \
  --run_bayesian_bambi \
  --no-run_bayesian_pymc

echo ""
echo "Example 1 complete! Check results in: $PTH_OUT_ROOT/CVD_three_sets_comparison_ortho_vi/"

# Example 2: Compare two sets with custom RDS parameters
echo ""
echo "Example 2: Comparing two sets with custom prefixes/suffixes"
python multi_prs_set_comparison_pipeline.py \
  --pth_covars $PTH_COVARS \
  --pth_dis $PTH_DIS \
  --pth_out_root $PTH_OUT_ROOT \
  --pth_out_dir CVD_custom_params \
  --prs_set_name DiffAE_Custom \
  --prs_set_root "/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc" \
  --rds_pres_prefix "run_ext_basic_king0p0625_lw_gw_indep_FiltMAF_" \
  --rds_pres_suffix ".fullDS.auto.mod.LDPred2.rds" \
  --prs_set_name External_Method \
  --prs_set_root "/path/to/external_rds_root" \
  --rds_pres_prefix "different_prefix_" \
  --rds_pres_suffix ".different_model.rds" \
  --reference_set DiffAE_Custom \
  --rds_tag_prs auto.mod \
  --tag_data resNdata.basic \
  --tag_prs pred_auto \
  --orthogonalise \
  --use_variational_inference

echo ""
echo "Example 2 complete! Check results in: $PTH_OUT_ROOT/CVD_custom_params_ortho_vi/"

# Example 3: Minimal comparison with only Elastic Net (fastest)
echo ""
echo "Example 3: Quick comparison with only Elastic Net method"
python multi_prs_set_comparison_pipeline.py \
  --pth_covars $PTH_COVARS \
  --pth_dis $PTH_DIS \
  --pth_out_root $PTH_OUT_ROOT \
  --pth_out_dir CVD_quick_elastic_net \
  --prs_set_name Set_A \
  --prs_set_root "/path/to/set_a_rds_root" \
  --prs_set_name Set_B \
  --prs_set_root "/path/to/set_b_rds_root" \
  --reference_set Set_A \
  --no-run_best_single \
  --run_elastic_net \
  --no-run_pca \
  --no-run_bayesian_pymc \
  --no-run_bayesian_bambi \
  --orthogonalise

echo ""
echo "Example 3 complete! Check results in: $PTH_OUT_ROOT/CVD_quick_elastic_net_ortho_mcmc/"

# Example 4: Comprehensive comparison with MCMC (most accurate, slowest)
echo ""
echo "Example 4: Comprehensive comparison with MCMC (will take several hours)"
echo "WARNING: This will take 5-9 hours to complete!"
read -P "Continue? (y/n): " confirm

if test "$confirm" = "y"
    python multi_prs_set_comparison_pipeline.py \
      --pth_covars $PTH_COVARS \
      --pth_dis $PTH_DIS \
      --pth_out_root $PTH_OUT_ROOT \
      --pth_out_dir CVD_comprehensive_mcmc \
      --prs_set_name DiffAE_Custom \
      --prs_set_root "/group/glastonbury/soumick/PRS/LDPred2/F20208v3_DiffAE/nonDisc" \
      --prs_set_name Comparison_Set \
      --prs_set_root "/path/to/comparison_rds_root" \
      --reference_set DiffAE_Custom \
      --no-use_variational_inference \
      --orthogonalise \
      --run_best_single \
      --run_elastic_net \
      --run_pca \
      --run_bayesian_pymc \
      --run_bayesian_bambi
    
    echo ""
    echo "Example 4 complete! Check results in: $PTH_OUT_ROOT/CVD_comprehensive_mcmc_ortho_mcmc/"
else
    echo "Example 4 skipped."
end

echo ""
echo "All examples complete!"
echo "Review the README at: MULTI_PRS_SET_PIPELINE_README.md"
