#!/usr/bin/env Rscript
# extract_scoring_files.R
# ------------------------------------------------------------------
# For each latent's `.fullDS.auto.mod.LDPred2.rds`, reconstruct the
# per-SNP map that corresponds to the LDpred2-auto `final_beta_auto`
# vector and write a "raw" PGS Catalog scoring TSV (no header yet).
#
# Reproduces the matching logic from
#   prs_pipeline/scripts/ldpred2_basic_ext_fullUKB.r
# but does NOT load the genotype matrix - we only need the BGEN map
# (from the .bgi index) plus the per-latent sumstats and the
# per-latent rsids2include filter file.
#
# Output columns (tab-separated, no PGS header):
#   chr_name, chr_position, rsID, effect_allele, other_allele, effect_weight
# ------------------------------------------------------------------

suppressPackageStartupMessages({
  library(optparse)
  library(bigsnpr)
  library(data.table)
  library(rjson)
})

option_list <- list(
  make_option(c('--rds_dir'), type = 'character',
              help = 'Directory containing *.fullDS.auto.mod.LDPred2.rds files'),
  make_option(c('--rds_prefix'), type = 'character',
              default = 'run_ext_basic_king0p0625_lw_gw_indep_FiltMAF_',
              help = 'Prefix of the RDS files for the paper run'),
  make_option(c('--rds_suffix'), type = 'character',
              default = '.fullDS.auto.mod.LDPred2.rds',
              help = 'Suffix of the auto-mod RDS files'),
  make_option(c('--bgen'), type = 'character',
              help = 'Path to the BGEN file used during PRS training (its .bgi index is read)'),
  make_option(c('--sumstats_root'), type = 'character',
              help = 'Directory containing per-latent {latent}.gwas.regenie.gz files'),
  make_option(c('--sumstats_cols'), type = 'character',
              default = '/group/glastonbury/soumick/MyCodes/GitLab/tricorder/PRS/sumcols_UKBB_regenie.json',
              help = 'JSON mapping of sumstats column names (UKBB regenie)'),
  make_option(c('--rsids_root'), type = 'character',
              help = 'Directory containing per-latent {latent}.txt rsIDs-to-include files'),
  make_option(c('--filt_maf'), type = 'numeric', default = 1,
              help = 'If 1, drop sumstats SNPs with MAF<0.01 (matches training)'),
  make_option(c('--output_dir'), type = 'character',
              help = 'Output directory for raw scoring TSVs'),
  make_option(c('--threads'), type = 'integer', default = 4),
  make_option(c('--only_latent'), type = 'character', default = NA,
              help = 'Optional: process only this latent ID (debug)')
)
opt <- parse_args(OptionParser(option_list = option_list))

stopifnot(!is.null(opt$rds_dir), !is.null(opt$bgen),
          !is.null(opt$sumstats_root), !is.null(opt$rsids_root),
          !is.null(opt$output_dir))

dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

# ---- load BGEN map (once) ----
message('[', Sys.time(), '] Loading BGEN .bgi index...')
bgi_file <- sub('\\.bgen$', '.bgen.bgi', opt$bgen)
bgi <- snp_readBGI(bgi_file, snp_id = NULL)
# Columns: chromosome, position, rsid, allele1, allele2 (allele1 = effect/alt in BGEN convention used here)
map_full <- data.frame(
  chr  = as.integer(bgi$chromosome),
  rsid = bgi$rsid,
  pos  = as.integer(bgi$position),
  a1   = bgi$allele1,
  a0   = bgi$allele2,
  stringsAsFactors = FALSE
)

# Helper: extract clean rsID from REGENIE-style IDs (e.g. rs199745162_A_C -> rs199745162).
# Non-rs IDs (e.g. 1:82133_CAA_C_CAA_C) return an empty string;
# chr_name + chr_position already provide the genomic coordinates.
clean_rsid <- function(x) {
  ifelse(grepl("^rs[0-9]+", x), sub("^(rs[0-9]+).*", "\\1", x), "")
}
message('[', Sys.time(), '] BGEN map: ', nrow(map_full), ' variants')

# ---- sumstats column mapping ----
jfile <- fromJSON(file = opt$sumstats_cols)
# Drop trailing flag entry "is_beta_precomp"
flag_key <- 'is_beta_precomp'
if (flag_key %in% names(jfile)) jfile[[flag_key]] <- NULL
canonical <- names(jfile)
file_cols <- as.character(jfile)

# ---- enumerate latents ----
rds_files <- list.files(opt$rds_dir,
                        pattern = paste0('^', gsub('([\\^\\$\\.\\|\\?\\*\\+\\(\\)\\[\\]\\{\\}\\\\])', '\\\\\\1', opt$rds_prefix),
                                         '.*', gsub('([\\^\\$\\.\\|\\?\\*\\+\\(\\)\\[\\]\\{\\}\\\\])', '\\\\\\1', opt$rds_suffix), '$'),
                        full.names = TRUE)
stopifnot(length(rds_files) > 0)
latents <- sub(paste0('^', opt$rds_prefix), '',
               sub(paste0(opt$rds_suffix, '$'), '', basename(rds_files)))

if (!is.na(opt$only_latent)) {
  keep <- latents == opt$only_latent
  rds_files <- rds_files[keep]
  latents <- latents[keep]
}
message('[', Sys.time(), '] Processing ', length(latents), ' latents')

# ---- loop ----
summary_rows <- data.frame()

for (i in seq_along(latents)) {
  latent <- latents[i]
  rds_path <- rds_files[i]
  out_tsv <- file.path(opt$output_dir, paste0(latent, '.scoring.raw.tsv'))

  message('\n[', Sys.time(), '] === ', latent, ' (', i, '/', length(latents), ') ===')

  if (file.exists(out_tsv)) {
    message('  Output exists, skipping: ', out_tsv)
    next
  }

  # 1. Load final_beta_auto
  mod <- readRDS(rds_path)
  if (!'final_beta_auto' %in% names(mod)) {
    warning('final_beta_auto missing in ', rds_path); next
  }
  fba <- mod$final_beta_auto
  message('  final_beta_auto length: ', length(fba),
          ', nonzero: ', sum(fba != 0))
  rm(mod); gc(verbose = FALSE)

  # 2. Per-latent rsids2include filter
  rsids_file <- file.path(opt$rsids_root, paste0(latent, '.txt'))
  if (!file.exists(rsids_file)) {
    warning('rsids2include missing: ', rsids_file); next
  }
  filt_rsids <- readLines(rsids_file)
  map_lat <- map_full[map_full$rsid %in% filt_rsids, ]
  message('  map filtered by rsids2include: ', nrow(map_lat), ' variants')

  # 3. Load sumstats
  ss_file <- file.path(opt$sumstats_root, paste0(latent, '.gwas.regenie.gz'))
  if (!file.exists(ss_file)) {
    warning('sumstats missing: ', ss_file); next
  }
  ss_raw <- fread(ss_file, select = file_cols, nThread = opt$threads,
                  showProgress = FALSE, data.table = FALSE)
  colnames(ss_raw) <- canonical
  ss_raw$chr <- as.integer(ss_raw$chr)
  ss_raw$pos <- as.integer(ss_raw$pos)

  # 4. Replicate training pipeline ordering:
  #    matched_indices <- match(rsIDs, sumstats$rsid)
  #    sumstats_ordered <- sumstats[na.omit(matched_indices), ]
  matched_indices <- match(map_lat$rsid, ss_raw$rsid)
  sumstats_ordered <- ss_raw[na.omit(matched_indices), ]

  # 5. MAF filter
  if (opt$filt_maf == 1) {
    sumstats_ordered$maf <- pmin(sumstats_ordered$a1freq, 1 - sumstats_ordered$a1freq)
    sumstats_ordered <- subset(sumstats_ordered, maf > 0.01)
  }

  # 6. snp_match against the BGEN map (use map_lat to mirror the rsid-filtered map)
  info_snp <- snp_match(sumstats_ordered, map_lat, join_by_pos = FALSE)
  message('  snp_match result: ', nrow(info_snp), ' variants')

  if (nrow(info_snp) != length(fba)) {
    stop('Length mismatch: info_snp=', nrow(info_snp),
         ' vs final_beta_auto=', length(fba),
         ' for latent ', latent,
         '. Cannot reconstruct PGS file safely.')
  }

  # 7. Build PGS Catalog raw scoring frame (in info_snp ordering)
  pgs <- data.frame(
    chr_name      = info_snp$chr,
    chr_position  = info_snp$pos,
    rsID          = clean_rsid(info_snp$rsid),
    effect_allele = info_snp$a1,
    other_allele  = info_snp$a0,
    effect_weight = fba,
    stringsAsFactors = FALSE
  )

  # 8. Drop zero-weight variants (LDpred2-auto truncated chains -> exact zero)
  nz <- pgs$effect_weight != 0 & !is.na(pgs$effect_weight)
  message('  retaining ', sum(nz), '/', nrow(pgs), ' non-zero variants')
  pgs <- pgs[nz, ]

  # 9. Sort by chr then pos for tidiness
  pgs <- pgs[order(pgs$chr_name, pgs$chr_position), ]

  fwrite(pgs, out_tsv, sep = '\t', quote = FALSE)
  message('  wrote ', out_tsv)

  summary_rows <- rbind(summary_rows, data.frame(
    latent = latent,
    n_variants = nrow(pgs),
    stringsAsFactors = FALSE
  ))
}

# ---- write summary ----
sumfile <- file.path(opt$output_dir, '_extract_summary.tsv')
fwrite(summary_rows, sumfile, sep = '\t', quote = FALSE)
message('\n[', Sys.time(), '] Summary -> ', sumfile)
message('Done.')
