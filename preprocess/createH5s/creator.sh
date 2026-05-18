#!/bin/bash

echo "Command executed: $0 $@"

echo "-----------------------"

#creation of argument helper
programmename=$0
function usage {
    echo ""
    echo "Launches an HDF5 creation proces."
    echo ""
    echo "usage: $programmename --hdf5_ready bool --run_tag string --tricorder_root string --creator_programme string --in_root string --in_dir string --out_path string --toyout_path string --dsV string --dir_unsorted bool --fID string --fDirName string --include string --filetag string --ds_names_present string string --argsH5 \"string\" or \" \""
    echo ""
    echo "  --run_tag           string    A tag to identify the run (Default: Blank)"
    echo "  --tricorder_root    string    Root directory of the Tricorder repo (Default: /group/glastonbury/soumick/MyCodes/GitLab/tricorder/)"
    echo "  --creator_programme string    Python file to run to create the HDF5 dataset, relative path inside the Tricorder repo (Default: Blank)"
    echo "  --in_root           string    Root directory of the input data (Default: /processing_data/shared_datasets/ukbiobank/raw_data/phenotypes/imaging)"
    echo "  --in_dir            string    Directory containing the input data (Default: Blank)"
    echo "  --out_path          string    Root directory of the output data (Default: /scratch/glastonbury/datasets/ukbbH5s)"
    echo "  --toyout_path       string    Root directory of the toy dataset  (Default: /group/glastonbury/soumick/toysets)"
    echo "  --dsV               string    Version of the dataset (Default: 3)"
    echo "  --dir_unsorted        0/1     Whether the input directory is unsorted (Default: 0)"
    echo "  --fID               string    fieldID of the bulk files to process, if dir_unsorted is 1 (Default: Blank)"
    echo "  --fDirName          string    Name of the directory for the particular filed, if dir_unsorted is 1 (Default: Blank)"
    echo "  --include           string    Comma-spereated list of main files to be included, without the extension. Leave it blank for all files mentioned inside the meta.yaml present inside the Tricorder (Default: Blank)"
    echo "  --filetag           string    File tag to be added to the name of the HDF5 file, only if there are multiple values in include argument, this will be used (Default: Blank)"
    echo "  --ds_names_present  string    Names of the datasets present in the HDF5 file, to be used for the post-processing steps after the creation of the HDF5 file. (Default: Blank)"
    echo "  --argsH5            string    Additional arguments for the HDF5 creation programme (Default: Blank)"
    echo ""
    echo "  Flags:"
    echo "  --hdf5_ready          0/1     Whether the HDF5 is already prepared, so the creation step will be skipped (Default: 0)"
    echo "  --create_splits       0/1     Whether to create the training splits (Default: 1)"
    echo "  --create_toydataset   0/1     Whether to create the toy dataset (Default: 1)"
    echo "  --get_metadata        0/1     Whether to get different metadata (modes 0, 2, 3, 5, 6, 7) from the HDF5 file (Default: 1)"
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

##################### set the default values for the commandline arguments
# run_tag="${run_tag:-}"
# tricorder_root="${tricorder_root:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder/}"
# creator_programme="${creator_programme:-}"
# in_root="${in_root:-/processing_data/shared_datasets/ukbiobank/raw_data/phenotypes/imaging}"
# in_dir="${in_dir:-}"
# out_path="${out_path:-/scratch/glastonbury/datasets/ukbbH5s}"
# toyout_path="${out_path:-/group/glastonbury/soumick/toysets}"
# dsV="${dsV:-3}"
# dir_unsorted="${dir_unsorted:-0}"
# fID="${fID:-}"
# fDirName="${fDirName:-}"
# include="${include:-}"
# filetag="${filetag:-}"
# ds_names_present="${ds_names_present:-}"
# argsH5="${argsH5:-}"

run_tag="${run_tag:-201}"
tricorder_root="${tricorder_root:-/group/glastonbury/soumick/MyCodes/GitLab/tricorder/}"
creator_programme="${creator_programme:-}"
in_root="${in_root:-/processing_data/shared_datasets/ukbiobank/raw_data/phenotypes/imaging}"
in_dir="${in_dir:-}"
out_path="${out_path:-/scratch/glastonbury/datasets/ukbbH5s}"
covar_out_path="${covar_out_path:-/group/glastonbury/GWAS/inputs/covariates}"
toyout_path="${toyout_path:-/group/glastonbury/soumick/toysets}"
dsV="${dsV:-3}"
dir_unsorted="${dir_unsorted:-0}"
fID="${fID:-}"
fDirName="${fDirName:-F20201_Dixon_technique_internal_fat_DICOM_RECOHPipe}"
include="${include:-}"
filetag="${filetag:-}"
ds_names_present="${ds_names_present:-primary_InOpp_0}"
argsH5="${argsH5:-}"

#covar-related params
cov_drop_MRICentre="${cov_drop_MRICentre:-0}"
cov_drop_duplicates="${cov_drop_duplicates:-1}"
cov_drop_smoking="${cov_drop_smoking:-1}"
cov_compute_BSA="${cov_compute_BSA:-0}"
cov_pth_outliers="${cov_pth_outliers:-}"

# flags
hdf5_ready="${hdf5_ready:-1}"
create_splits="${create_splits:-1}"
create_toydataset="${create_toydataset:-1}"
get_metadata="${get_metadata:-1}"
create_covars="${create_covars:-1}"

##################### create / check the HDF5 file
in_path="${in_root}/${in_dir}"

if [ "$hdf5_ready" -eq 0 ]; then
  # crH5job=$(sbatch -J cr${run_tag} /home/soumick.chatterjee/backupcpurun.sh --root ${tricorder_root} --programme ${creator_programme} --args "--in_path ${in_path} --out_path ${out_path} --dsV ${dsV} $( [ "$dir_unsorted" -eq 1 ] && echo "--dir_unsorted" ) --fID ${fID} --fDirName ${fDirName} --include ${include} --filetag ${filetag}" ${argsH5} | awk '{print $NF}')
  crH5job=$(sbatch -J cr${run_tag} /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme ${creator_programme} --args "$( [ -n "$in_path" ] && echo "--in_path $in_path" ) $( [ -n "$out_path" ] && echo "--out_path $out_path" ) $( [ -n "$dsV" ] && echo "--dsV $dsV" ) $( [ "$dir_unsorted" -eq 1 ] && echo "--dir_unsorted" || echo "--no-dir_unsorted" ) $( [ -n "$fID" ] && echo "--fID $fID" ) $( [ -n "$fDirName" ] && echo "--fDirName $fDirName" ) $( [ -n "$include" ] && echo "--include $include" ) $( [ -n "$filetag" ] && echo "--filetag $filetag" ) ${argsH5}" | awk '{print $NF}')
  echo "HDF5 creation job: ${crH5job}"
  jobdepend="--dependency=afterok:${crH5job}"
else
  echo "HDF5 is already prepared, so the creation step will be skipped."
  jobdepend=""
fi

if [ "$dir_unsorted" -eq 0 ] && [ -z "$fDirName" ]; then
  fDirName=$(basename "$in_path")
fi

# Construct the output path
full_out_path="${out_path}/${fDirName}_H5"

#deal with dsV
if [ -n "$dsV" ]; then
  if [[ "${dsV,,}" == *v* ]]; then
    full_out_path="${full_out_path}${dsV,,}"  # Append the lowercase dsV directly, if "v" is present in the name
  else
    full_out_path="${full_out_path}v${dsV}"  # Append "v" followed by the supplied value of dsV
  fi
fi

h5file="${full_out_path}/data${filetag}.h5"

if [ "$hdf5_ready" -eq 0 ]; then
  echo "HDF5 that will be produced by this script: ${h5file}"
else
  if [ ! -f "$h5file" ]; then
    die "HDF5 file does not exist: ${h5file}"
  else
    echo "HDF5 file to be processed: ${h5file}"
  fi
fi

##################### create the splits
if [ "$create_splits" -eq 1 ]; then
echo "Training splits will be created..."
  sbatch $jobdepend -J splt${run_tag} /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme preprocess/dsSpliter.py --args "--patient_n_sessions 1 $( [ -n "$ds_names_present" ] && echo "--ds_names_present $ds_names_present" ) $( [ -n "$h5file" ] && echo "--path_h5s $h5file" )"
  sbatch $jobdepend -J splt${run_tag} /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme preprocess/dsSpliter.py --args "--patient_n_sessions 0 $( [ -n "$ds_names_present" ] && echo "--ds_names_present $ds_names_present" ) $( [ -n "$h5file" ] && echo "--path_h5s $h5file" )"
fi

##################### create the toydataset
if [ "$create_toydataset" -eq 1 ]; then  
  toy_h5file="${h5file/$out_path/$toyout_path}" # Replace the out_path with toyout_path
  base_dir=$(dirname "$toy_h5file")                        # Get the directory part of the path
  last_folder=$(basename "$base_dir")                      # Get the last folder name
  new_base_dir=$(dirname "$base_dir")/dummy_"$last_folder" # Add 'dummy_' to the last folder
  toy_h5file="$new_base_dir/$(basename "$toy_h5file")"         # Combine with the file name
  echo "Toy dataset will be created as: ${toy_h5file}"
  sbatch $jobdepend -J toy${run_tag} /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme H5tools/create_toyset.py --args "$( [ -n "$h5file" ] && echo "--in_path $h5file" ) $( [ -n "$toy_h5file" ] && echo "--out_path $toy_h5file" ) $( [ -n "$ds_names_present" ] && echo "--ds_names_present $ds_names_present" )"
fi


##################### get different metadata from the HDF5 file
if [ "$get_metadata" -eq 1 ]; then
  echo "Metadata (modes 0, 2, 3, 5, 6, 7) will be extracted..."
  h5_dir_path=$(dirname "$h5file")
  h5_file_name=$(basename "$h5file")
  #subID
  subIDjob=$(sbatch $jobdepend -J me0ta${run_tag} /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme H5tools/traverse_DSH5.py --args "$( [ -n "$h5_dir_path" ] && echo "--in_path $h5_dir_path" ) $( [ -n "$h5_file_name" ] && echo "--dataH5 $h5_file_name" ) --mode 0 $( [ -n "$ds_names_present" ] && echo "--ds_names_present $ds_names_present" )" | awk '{print $NF}') 
  #mean+STD
  sbatch $jobdepend -J me2ta${run_tag} /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme H5tools/traverse_DSH5.py --args "$( [ -n "$h5_dir_path" ] && echo "--in_path $h5_dir_path" ) $( [ -n "$h5_file_name" ] && echo "--dataH5 $h5_file_name" ) --mode 2 $( [ -n "$ds_names_present" ] && echo "--ds_names_present $ds_names_present" )" 
  #unique DS and counts
  sbatch $jobdepend -J me3ta${run_tag} /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme H5tools/traverse_DSH5.py --args "$( [ -n "$h5_dir_path" ] && echo "--in_path $h5_dir_path" ) $( [ -n "$h5_file_name" ] && echo "--dataH5 $h5_file_name" ) --mode 3 $( [ -n "$ds_names_present" ] && echo "--ds_names_present $ds_names_present" )" 
  #MRI date
  datejob=$(sbatch $jobdepend -J me5ta${run_tag} /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme H5tools/traverse_DSH5.py --args "$( [ -n "$h5_dir_path" ] && echo "--in_path $h5_dir_path" ) $( [ -n "$h5_file_name" ] && echo "--dataH5 $h5_file_name" ) --mode 5 $( [ -n "$ds_names_present" ] && echo "--ds_names_present $ds_names_present" )" | awk '{print $NF}') 
  #MRI centre
  centrejob=$(sbatch $jobdepend -J me6ta${run_tag} /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme H5tools/traverse_DSH5.py --args "$( [ -n "$h5_dir_path" ] && echo "--in_path $h5_dir_path" ) $( [ -n "$h5_file_name" ] && echo "--dataH5 $h5_file_name" ) --mode 6 $( [ -n "$ds_names_present" ] && echo "--ds_names_present $ds_names_present" )" | awk '{print $NF}')
  #unique shapes
  sbatch $jobdepend -J me7ta${run_tag} /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme H5tools/traverse_DSH5.py --args "$( [ -n "$h5_dir_path" ] && echo "--in_path $h5_dir_path" ) $( [ -n "$h5_file_name" ] && echo "--dataH5 $h5_file_name" ) --mode 7 $( [ -n "$ds_names_present" ] && echo "--ds_names_present $ds_names_present" )" 
fi

##################### create the covariates
if [ "$create_covars" -eq 1 ]; then
  echo "Covariates will be created..."
  if [ "$get_metadata" -eq 1 ]; then
    jobdepend="--dependency=afterok:${subIDjob}"
    if [ "$cov_drop_MRICentre" -eq 0 ]; then
      jobdepend="${jobdepend}:${centrejob}"
    fi
  else
    jobdepend=""
  fi
  echo "dependencies: ${jobdepend}"

  base_dir=$(dirname "$h5file") 
  if [[ -n "$ds_names_present" ]]; then
      # Split ds_names_present into a list
      if [[ "$ds_names_present" == *","* ]]; then
          IFS=',' read -ra ds_names_present_list <<< "$ds_names_present"
      else
          IFS=$'\n' read -d '' -r -a ds_names_present_list < <(echo "$ds_names_present" | awk -v RS='OR' '{print $0}')
      fi
      
      # Check the length of the list
      if [[ "${#ds_names_present_list[@]}" -eq 1 ]]; then
          tag="_${ds_names_present_list[0]}"
      else
          tag="_$(printf "%sOR" "${ds_names_present_list[@]}")"
          tag=${tag%OR}
      fi
  else
      tag=""
  fi
  echo "tag: ${tag}"

  if [ -n "$filetag" ]; then
    filetag="data${filetag}_"
  fi
  echo "filetag: ${filetag}"

  sbatch $jobdepend -J cov${run_tag} /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme GWAS/create_covariates.py --args "--pth_prefix $base_dir/meta --pth_ids ${filetag}subIDs${tag}.json --pth_MRIdates ${filetag}subIDs_MRIdates${tag}.csv --pth_MRIcentre ${filetag}subIDs_Acqs_MRICentre${tag}.json --no-whole_cohort --outpth ${covar_out_path} $( [ -n "$ds_names_present" ] && echo "--data_keys $ds_names_present" ) $( [ "$cov_drop_MRICentre" -eq 1 ] && echo "--drop_MRICentre" || echo "--no-drop_MRICentre" ) $( [ "$cov_drop_duplicates" -eq 1 ] && echo "--drop_duplicates" || echo "--no-drop_duplicates" ) $( [ "$cov_drop_smoking" -eq 1 ] && echo "--drop_smoking" || echo "--no-drop_smoking" ) $( [ "$cov_compute_BSA" -eq 1 ] && echo "--compute_BSA" || echo "--no-compute_BSA" ) $( [ -n "$cov_pth_outliers" ] && echo "--pth_outliers $cov_pth_outliers" )"
  if [ "$cov_drop_MRICentre" -eq 0 ]; then
    sbatch $jobdepend -J cov${run_tag}NoCen /home/soumick.chatterjee/cpurun.sh --root ${tricorder_root} --programme GWAS/create_covariates.py --args "--pth_prefix $base_dir/meta --pth_ids ${filetag}subIDs${tag}.json --pth_MRIdates ${filetag}subIDs_MRIdates${tag}.csv --pth_MRIcentre ${filetag}subIDs_Acqs_MRICentre${tag}.json --no-whole_cohort --outpth ${covar_out_path} $( [ -n "$ds_names_present" ] && echo "--data_keys $ds_names_present" ) --drop_MRICentre $( [ "$cov_drop_duplicates" -eq 1 ] && echo "--drop_duplicates" || echo "--no-drop_duplicates" ) $( [ "$cov_drop_smoking" -eq 1 ] && echo "--drop_smoking" || echo "--no-drop_smoking" ) $( [ "$cov_compute_BSA" -eq 1 ] && echo "--compute_BSA" || echo "--no-compute_BSA" ) $( [ -n "$cov_pth_outliers" ] && echo "--pth_outliers $cov_pth_outliers" )"
  fi
fi