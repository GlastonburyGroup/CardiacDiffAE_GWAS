#Step0 [To be done only once]: LD Pruning with PLINK2 window size 50, step 5, R^2 threshold 0.5 (LDPruning.sh). This generates a pruned list of SNPs with .prune.in extension
#Step0 [Alternate option, to be done only once]: Create a list of SNPS from the hapmap3+ data
#Step1: Create a list of SNPS that are relavent for us.
#######Step1.1: List of GNOME-wide significant SNPs obtained after running GWAS 
#######Step1.2: List of SNPs obtained after running conditional analysis via Cojo on the GWAS summary statistics
#######For our discovery cohort, we have them here: /group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/results/gwas/independent/genome-wide_significant_hits_post_cojo.csv
#Step2: Subset the bgen file using the list of SNPs obtained in Step0 and Step1 (subset_bgen.sh)
#Step3 [Optional, only if we are using sumstats-based methods]: Subset the summary statistics file using the list of SNPs obtained in Step0 and Step1 (subset_sumstats.sh)
