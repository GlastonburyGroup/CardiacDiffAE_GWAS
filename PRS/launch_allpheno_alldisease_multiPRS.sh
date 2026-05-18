#!/bin/bash

default_config_dir="/scratch/soumick.chatterjee/tmp"

# Usage function
usage() {
  echo "Usage: $0 --key0 value0 --key1 value1  ... --config-dir /path/to/configs"
  exit 1
}

# Declare an associative array to store the arguments
declare -A args
config_dir=""

# Parse command-line arguments
while (( "$#" )); do
  if [[ "$1" == --* ]]; then
    key="${1#--}"
    shift
    value="$1"
    if [[ "$key" == "config-dir" ]]; then
      config_dir="$value"
    else
      if [ -z "$key" ] || [ -z "$value" ]; then
        usage
      fi
      args["$key"]="$value"
    fi
    shift
  else
    usage
  fi
done

# Use the provided config directory or default if not provided
if [ -z "$config_dir" ]; then
  config_dir="$default_config_dir"
fi

# Create the config directory if it doesn't exist
mkdir -p "$config_dir"

# Generate a unique config file for this job
config_file=$(mktemp "$config_dir/config_multiPRS_XXXXXXXXXXXXXXX")
for key in "${!args[@]}"; do
  echo "$key=${args[$key]}" >> "$config_file"
done

echo "Configuration file created at: $config_file"

# # Submit the first job and capture its job ID
jobid0=$(sbatch --exclude=cnode01 --parsable --array=0 --export=config_file="$config_file" /group/glastonbury/soumick/MyCodes/GitLab/tricorder/PRS/allpheno_alldisease_multiPRSV2_panCohort.sh)

# # Submit the array job with a dependency on the first job
jobid_array=$(sbatch --exclude=cnode01 --parsable --array=1-18 --export=config_file="$config_file" --dependency=afterok:$jobid0 /group/glastonbury/soumick/MyCodes/GitLab/tricorder/PRS/allpheno_alldisease_multiPRSV2_panCohort.sh)
# jobid_array=$(sbatch --parsable --array=0-18 --export=config_file="$config_file" /group/glastonbury/soumick/MyCodes/GitLab/tricorder/PRS/allpheno_alldisease_multiPRSV2_panCohort.sh)

# # Submit the final job for evaluating and plotting the results
sbatch --exclude=cnode01 --export=config_file="$config_file" --dependency=afterok:$jobid_array /group/glastonbury/soumick/MyCodes/GitLab/tricorder/PRS/allpheno_alldisease_multiPRS_panCohort_eval.sh
# sbatch --export=config_file="$config_file" /group/glastonbury/soumick/MyCodes/GitLab/tricorder/PRS/allpheno_alldisease_multiPRS_panCohort_eval.sh