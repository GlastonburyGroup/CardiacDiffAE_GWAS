library(optparse)
#install.packages("optparse")

option_list = list(
  make_option(c('--root_clinical_data'), action='store', type='character', help='Where the clinical files are storred', default='../Processed_clinical_data/'),
  make_option(c('--pth_cov'), action='store', type='character', help='Path the covariates TSV file for the considered cohort (discovery+replication)', default='/group/glastonbury/GWAS/F20208v3_DiffAE/select_latents_r80/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/nNs_Qntl_INF30_DiffAE128_5Sd_r80_discov_fullDSV3/validated_input/cov_newset_chp_F20208_Long_axis_heart_images_DICOM_H5v3_NOnoise.cov.validated.txt'),
  make_option(c('--pth_cov_wholecohort'), action='store', type='character', help='Path the covariates TSV file for the whole cohort', default='../PRS/inputs/cov_nonMRI_cohort.tsv'),
  make_option(c('--pth_subIDs_discov'), action='store', type='character', help='Path the coma-seperated file containing the list of subIDs in the discovery cohort', default='/group/glastonbury/GWAS/F20208v3_DiffAE/subIDs_MRI_discovery_V2WBRIT.txt'),
  make_option(c('--pth_subIDs_replic'), action='store', type='character', help='Path the coma-seperated file containing the list of subIDs in the replication cohort (leave it blank if not available)', default='/group/glastonbury/GWAS/F20208v3_DiffAE/subIDs_MRI_replication_V2nonWBRIT_OnlyV3.txt'),
  make_option(c('--root_out'), action='store', type='character', help='', default='../clinicaldata/binary_disease_cohorts/disease_files/'),
  make_option(c('--out_filename'), action='store', type='character', help='', default='cardiac_SR_HI_GP_v3MRI_allUKBB')
)

opt = parse_args(OptionParser(option_list=option_list))

setwd(opt$root_clinical_data)

icd10Map <- read.table("mapping_tables/icd10_map.tsv", sep="\t", header=T, quote="\"" )
icd9Map <- read.table("mapping_tables/icd9_map.tsv", sep="\t", header=T, quote="\"" )

convert.ICD10 <- function(df, field)
{
  icd10 <- df[df[, field[2]] == "icd_10", ]
  icd10[, field[1]] <- sapply(icd10[, field[1]], gsub, pattern="X$", replacement="")
  icd10 <- merge(icd10, icd10Map, by.x=field[1], by.y="coding", all.x=TRUE, all.y=FALSE)
  
  icd9 <- df[df[, field[2]] == "icd_9", ]
  icd9 <- merge(icd9, icd9Map, by.x=field[1], by.y="coding", all.x=TRUE, all.y=FALSE)
  
  rbind(icd10, icd9)
}

cov <- read.table(opt$pth_cov, header=T, sep="\t")

dis <- as.integer(scan(opt$pth_subIDs_discov, what = character(), sep = ','))

rep <- NULL
if (opt$pth_subIDs_replic != '') {
  rep <- scan(opt$pth_subIDs_replic, what = character(), sep = ',')
}


#### new - add non MRI individuals
cohort.gwas <- intersect(cov$FID, c(dis, rep))
cov_all <- read.table(opt$pth_cov_wholecohort, header=T, sep="\t")
cohort.nonGWAS <- setdiff(cov_all$FID, cohort.gwas)

cohort.whole <- c(cohort.gwas, cohort.nonGWAS)
####


#Self reported
sr <- read.csv("SR_Non_Cancer_Illness_ALL.csv")
#sr <- sr[sr$eid %in% cov$FID, ]

sr_diseases <- c("angina", 
                 "atrial fibrillation", "atrial flutter",
                 "heart attack/myocardial infarction",
                 "high cholesterol", 
                 "hypertension", "essential hypertension",
                 "type 2 diabetes", 
                 "aortic aneurysm rupture", 
                 "aortic dissection", "aortic valve disease", 
                 "aortic aneurysm", "aortic regurgitation / incompetence", "aortic stenosis", 
                 "arterial embolism", 
                 "mitral stenosis", "mitral valve disease", "mitral valve prolapse", "mitral regurgitation / incompetence",
                 "pericarditis", "pericardial problem", "pericardial effusion", 
                 "myocarditis",
                 "heart/cardiac problem")

sr$source <- "SR"

sr.diseases.eid <- unique(sr$eid[sr$meaning %in% sr_diseases]) 

sr$summary <- NA
sr$summary[sr$meaning == "angina"] <- "angina pectoris"
sr$summary[sr$meaning %in% c("atrial fibrillation", "atrial flutter")] <- "atrial fibrillation/flutter"
sr$summary[sr$meaning == "heart attack/myocardial infarction"] <- "myocardial infarction"
sr$summary[sr$meaning == "high cholesterol"] <- "high cholesterol"
sr$summary[sr$meaning %in% c("hypertension", "essential hypertension")] <- "hypertension"
sr$summary[sr$meaning == "type 2 diabetes"] <- "type 2 diabetes"
sr$summary[sr$meaning %in% c("mitral stenosis", "mitral valve disease", "mitral valve prolapse", "mitral regurgitation / incompetence")] <- "mitral problems"
sr$summary[sr$meaning %in% c("pericarditis", "pericardial problem", "pericardial effusion")] <- "pericardial problem"
sr$summary[sr$meaning == "myocarditis"] <- "myocarditis"
sr$summary[sr$meaning == "heart/cardiac problem"] <- "heart/cardiac problem" 

# "heart failure/pulmonary odema", "stroke""

#GP diagnosed and hospital admission (ICD10 codes)
gp <- convert.ICD10(read.csv("GP_Clinical_ICD10_ALL.csv"), c("icd_code", "icd_type"))
#gp <- gp[gp$eid %in% cov$FID, ]

hi <- convert.ICD10(read.csv("HI_diagnosis_ALL.csv"), c("icd", "icd_version"))
#hi <- hi[hi$eid %in% cov$FID, ]

#All the ICD-codes can go togther
colnames(hi)[which(colnames(hi) == "icd_version")] <- "icd_type"
hi$source <- "HI"
colnames(gp)[which(colnames(gp) == "icd_code")] <- "icd"
gp$source <- "GP"

hi$diag_type <- NULL
icd <- rbind(gp, hi)

#I filter for the relevant diseases (using the ICD-chapters)
# For ICD-10
# 	^I chapthers are diseases of the CV system (including hypertension)
# 	^E11 is T2D
# 	^E78.0 is Pure hypercholesterolaemia
# For ICD-09
#  	^39, ^40, ^41, ^42, ^43, ^44, ^45  are diseases of the CV system (including hypertension)
#	^250, (^249 is secondary diabetes mellitus, I'm not considering this here)
#	^272.0 is Pure hypercholesterolaemia
icd <- icd[(icd$icd_type == "icd_10" & grepl("^I|^E11|^E78", icd$icd)) | (icd$icd_type == "icd_9" & grepl("^39|^40|^41|^42|^43|^44|^45|^272", icd$icd)), ]


#All of these have some kind of CVD and/or T2D and will not be included in the set of healthy individuals. These will be super controls
# I've some ICD9 that could not be mapped, so I'll discard them from both the diseased and the healthy (N record=182)
icd.diseased.eid <- unique(icd$eid) 


icd$summary <- NA 

icd$summary[grepl("E78.0|272.0", icd$meaning)] <- "high cholesterol" 
icd$summary[grepl("fibrillation", icd$meaning)] <- "atrial fibrillation/flutter" 
icd$summary[grepl("atrial", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "atrial fibrillation/flutter" 
icd$summary[grepl("angina", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "angina pectoris" 
icd$summary[grepl("mitral", icd$meaning, ignore.case=TRUE) & is.na(icd$summary) & !grepl("Rheumatic", icd$meaning, ignore.case=TRUE)] <- "mitral problems"
icd$summary[grepl("myocarditis", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "myocarditis" 
icd$summary[grepl("hypertension", icd$meaning, ignore.case=TRUE) & is.na(icd$summary) & !grepl("pulmonary|Renovascular|renal", icd$meaning, ignore.case=TRUE)] <- "hypertension"
icd$summary[grepl("E11|250", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "type 2 diabetes" 
icd$summary[grepl("myocardial", icd$meaning, ignore.case=TRUE) & is.na(icd$summary) & !grepl("following|not resulting|degeneration", icd$meaning, ignore.case=TRUE)] <- "myocardial infarction"
icd$summary[grepl("pericard", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "pericardial problem"
icd$summary[grepl("ischaemic heart disease|Atherosclerotic heart disease", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "coronary heart disease" #This include chronic and acute, and caused by the atherosclerotic plaques
icd$summary[grepl("cardiomyopathy|cardiomyopathies", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "cardiomyopathy"
icd$summary[grepl("I73.0", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "Raynaud's syndrome"
icd$summary[grepl("arrhythmia|tachycardia", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "arrhythmia/tachycardia"
icd$summary[grepl("Hypotension", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "hypotension"
icd$summary[grepl("Cardiomegaly", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "cardiomegaly"
icd$summary[grepl("bundle-branch", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "Tawara branches"
icd$summary[grepl("Aortic", icd$meaning, ignore.case=TRUE) & is.na(icd$summary) & !grepl("Rheumatic|aneurys", icd$meaning, ignore.case=TRUE)] <- "aortic valve disorders"
icd$summary[icd$meaning == "I35 Nonrheumatic aortic valve disorders" & is.na(icd$summary)] <- "aortic valve disorders"
icd$summary[grepl("depolarisation", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "premature depolarisation"
icd$summary[grepl("I50.1", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "left ventricular failure"
icd$summary[grepl("Atrioventricular", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "atrioventricular block"
icd$summary[grepl("I51.8|I51.9", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "other heart diseases"
icd$summary[icd$meaning == "I25.4 Coronary artery aneurysm" & is.na(icd$summary)] <- "coronary artery aneurysm"
icd$summary[grepl("Heart failure", icd$meaning, ignore.case=TRUE) & is.na(icd$summary) & !grepl("Hypertensive", icd$meaning, ignore.case=TRUE)] <- "heart failure"
icd$summary[icd$meaning == "I51.3 Intracardiac thrombosis, not elsewhere classified" & is.na(icd$summary)] <- "intracardiac thrombosis"
icd$summary[icd$meaning == "I82.2 Embolism and thrombosis of vena cava" & is.na(icd$summary)] <- "embolism and thrombosis of vena cava"
icd$summary[icd$meaning == "I25.3 Aneurysm of heart" & is.na(icd$summary)] <- "aneurysm of heart"
icd$summary[grepl("Thoracic aortic aneurysm", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "thoracic aortic aneurysm"
icd$summary[grepl("fascicular block", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "fascicular block"
icd$summary[grepl("Cardiac arrest", icd$meaning, ignore.case=TRUE) & is.na(icd$summary)] <- "cardiac arrest"


#Extracts the diseased individuals to use of association
sr <- sr[!is.na(sr$summary), c("eid", "date", "meaning", "summary", "source")]
icd <- icd[!is.na(icd$summary), c("eid", "date", "meaning", "summary", "source")]

diseased <- unique(rbind(icd, sr))

#add info on discovery/replication + MRI_date
diseased$year <- as.numeric(sapply(diseased$date, function(s) unlist(strsplit(s, split="-"))[1]))
diseased <- merge(diseased, cov[, c("IID", "MRI_Date")], by.x="eid", by.y="IID", all.x=T, all.y=F)


diseased$cohort <- ifelse(as.character(diseased$eid) %in% dis, "Discovery", "nonMRI")
diseased$cohort <- ifelse(as.character(diseased$eid) %in% rep, "Replication", diseased$cohort)

length(unique(diseased$eid))

#diseased <- diseased[!is.na(diseased$cohort), ]
diseased <- diseased[diseased$eid %in% cohort.whole, ]


setwd(opt$root_out)
save(diseased, file=paste0(opt$out_filename, "_disease.RData"))
write.csv(diseased, file = paste0(opt$out_filename, '_disease.csv'), row.names = FALSE)



#Extract the healthy individuals to use for association
healthy.ids <- unique(setdiff(cohort.whole, unique(diseased$eid)))
healthy <- data.frame(eid=healthy.ids, date=NA, meaning=NA, summary="healthy", source=NA, year=NA)

healthy <- merge(healthy, cov[, c("IID", "MRI_Date")], by.x="eid", by.y="IID", all.x=T, all.y=F)
healthy$cohort <- ifelse(as.character(healthy$eid) %in% dis, "Discovery", "nonMRI")
healthy$cohort <- ifelse(as.character(healthy$eid) %in% rep, "Replication", healthy$cohort)

length(healthy$eid)

df <- rbind(diseased, healthy)

length(unique(df$eid))
df <- df[df$eid %in% cohort.whole, ]


#write.csv(df, file = 'merge_SR_HI_GP_v3_&_HEALTHY.csv', row.names = FALSE)
write.csv(df, file = paste0(opt$out_filename, '_disease_N_control.csv'), row.names = FALSE)