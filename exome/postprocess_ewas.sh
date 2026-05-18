#!/bin/bash
#SBATCH --job-name §§JOBNAME§§
#SBATCH --mail-type=ALL
#SBATCH --mail-user=§§EMAIL§§
#SBATCH --chdir=§§WORKDIR§§
#SBATCH --output §§JOBNAME§§_%A.log
#SBATCH --partition cpuq
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=5000Mb
#SBATCH --time 3-00:00:00

module load nextflow/22.10.1 singularity/3.8.5

export NXF_OPTS="-Xms1G -Xmx4G"
export NXF_HOME=§§WORKDIR§§/.nextflow_§§JOBNAME§§

path2gwas="§§WORKDIR§§/§§JOBNAME§§_output"

#Run the GWAS pipeline
nextflow run §§PIPELINEROOT§§nf-pipeline-regenie \
   -profile singularity,ht_cluster_danger -c §§CONFFILE§§ §§NF_NODEEXCLD§§

if [[ $? == 0 ]]
then
   echo "NEXTFLOW SUCCESSFUL (most likely!). Launching gwasvis and ldtrait jobs!"

   #Create the QQ and Manhattan plots (with and without filtering based on the top loci)
   sbatch §§NODEEXCLUDE§§ /home/soumick.chatterjee/gwasvis.sh --args "--path2gwasout $path2gwas/§§JOBNAME§§ --res_type ewas --no-filter_alfreq --sig_level 5e-6 --n_ind_tests 30 --no-stratify_qq"

   #Create the combined toploci file
   lastJobID=$(sbatch  -J ovrlp §§NODEEXCLUDE§§ /home/soumick.chatterjee/cpurun.sh --root /group/glastonbury/soumick/MyCodes/GitLab/tricorder/ --programme GWAS/analyses/overlap.py --args "--path2gwasouts $path2gwas/§§JOBNAME§§" | awk '{print $NF}')

   #Fetch the LD traits from LDLink and create the wordclouds
   sbatch --dependency=afterok:$lastJobID §§NODEEXCLUDE§§ /home/soumick.chatterjee/ldtrait.sh --args "--toplocifile $path2gwas/§§JOBNAME§§"
else
   echo "NEXTFLOW FAILED! Checking if the sumstats were created. If yes, then we need to just obtain the toploci and continue with the work as usual!"

   phenotypes="§§PHENOLIST§§"
   IFS=',' read -r -a phenotypes <<< "$phenotypes"

   path2sumstats="$path2gwas/§§JOBNAME§§/results/gwas"
   sumstats=($(ls $path2sumstats/*.gwas.regenie.gz | xargs -n 1 basename))
   sumstats_tbi=($(ls $path2sumstats/*.gwas.regenie.gz.tbi | xargs -n 1 basename))   

   #check if each element of the phenotypes array is present in the sumstats and sumstats_tbi arrays
   missing_sumstats=0
   missing_sumstats_tbi=0
   for i in "${phenotypes[@]}"
   do
      if [[ ! " ${sumstats[@]} " =~ " ${i}.gwas.regenie.gz " ]]; then
         echo "$i.gwas.regenie.gz is missing"
         missing_sumstats=$((missing_sumstats+1))
      fi
      if [[ ! " ${sumstats_tbi[@]} " =~ " ${i}.gwas.regenie.gz.tbi " ]]; then
         echo "$i.gwas.regenie.gz.tbi is missing"
         missing_sumstats_tbi=$((missing_sumstats_tbi+1))
      fi
   done

   #if the number of missing sumstats and sumstats_tbi is 0, then we can proceed with the rest of the analysis
   if [[ $missing_sumstats -eq 0 ]] && [[ $missing_sumstats_tbi -eq 0 ]]
   then
      echo "All sumstats and sumstats_tbi files are present. That implies GWAS ran perfectly, only crashed during toploci/tophits generation."
      echo "Performing toploci creation using PLINK clumping, and then will continue with the rest of the anslysis!"

      #can launch the creation of the QQ and Manhattan plots (without filtering based on the top loci) without waiting for the toplociJob to finish
      sbatch §§NODEEXCLUDE§§ /home/soumick.chatterjee/gwasvis.sh --args "--path2gwasout $path2gwas/§§JOBNAME§§"

      #Create the toploci files using PLINK clumping
      toplociJob=$(sbatch §§NODEEXCLUDE§§ /group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/obtain_toploci.sh --phenos §§PHENOLIST§§ --indir $path2sumstats | awk '{print $NF}')

      #Create the QQ and Manhattan plots (with filtering based on the top loci)      
      sbatch --dependency=afterok:$toplociJob §§NODEEXCLUDE§§ /home/soumick.chatterjee/gwasvis.sh --args "--path2gwasout $path2gwas/§§JOBNAME§§ --filter_toploci_mode 1"

      #Create the combined toploci file
      lastJobID=$(sbatch --dependency=afterok:$toplociJob -J ovrlp §§NODEEXCLUDE§§ /home/soumick.chatterjee/cpurun.sh --root /group/glastonbury/soumick/MyCodes/GitLab/tricorder/ --programme GWAS/analyses/overlap.py --args "--path2gwasouts $path2gwas/§§JOBNAME§§" | awk '{print $NF}')

      #Fetch the LD traits from LDLink and create the wordclouds
      sbatch --dependency=afterok:$lastJobID §§NODEEXCLUDE§§ /home/soumick.chatterjee/ldtrait.sh --args "--toplocifile $path2gwas/§§JOBNAME§§"
   else
      echo "There are missing sumstats and sumstats_tbi files. Please check (maybe resume or re-run the GWAS!)"
      echo "missing number of sumstats: $missing_sumstats"
      echo "missing number of sumstats_tbi: $missing_sumstats_tbi"
   fi
fi