#!/bin/bash

# Exit script on any error. The -x flag prints each command, which is useful for debugging.
set -e -x -o pipefail

# Define mount point for clarity
MOUNT_POINT="/home/dnanexus/project_mount"

# --- Cleanup function to unmount the FUSE directory ---
# This will be called automatically when the script exits for any reason.
cleanup() {
    echo "--- Executing cleanup: Unmounting FUSE directory ---"
    # Check if the mount point is active before trying to unmount
    if mountpoint -q "${MOUNT_POINT}"; then
        fusermount -uz "${MOUNT_POINT}"
        echo "Unmount successful."
    else
        echo "Mount point not found or already unmounted."
    fi
}
# Trap ensures the cleanup function runs on script exit
trap cleanup EXIT

# --- Main Execution ---
echo "Starting H5 creation applet..."

# 0. Prepare the environment, install dependencies, and capture project name.

# Capture the project name in a variable
PROJECT_NAME=$(dx describe --json "$DX_PROJECT_CONTEXT_ID" | jq -r .name)
echo "Running in project: ${PROJECT_NAME}"

# # --- Manual installation of dxfuse ---
echo "Installing dxfuse manually..."
DXFUSE_VERSION="0.23.0"
# Download the binary, name it 'dxfuse', and place it in a directory on the system's PATH
wget "https://github.com/dnanexus/dxfuse/releases/download/v${DXFUSE_VERSION}/dxfuse-linux" -O /usr/local/bin/dxfuse
chmod +x /usr/local/bin/dxfuse
echo "dxfuse installed successfully at $(which dxfuse)."
# --- End of installation ---

# 1. Create directories for the FUSE mount point and for outputs.
mkdir -p "${MOUNT_POINT}"
mkdir -p /home/dnanexus/out

# 2. Mount the entire project using dxfuse.
echo "Mounting project ${DX_PROJECT_CONTEXT_ID} to ${MOUNT_POINT}..."
dxfuse "${MOUNT_POINT}" "${DX_PROJECT_CONTEXT_ID}"
echo "Mount command issued."

# 3. Construct the full path to the source data directory.
CLEANED_INPUT_PATH=$(echo "${in_path}" | sed 's#^/##')
SOURCE_DATA_DIR="${MOUNT_POINT}/${PROJECT_NAME}/${CLEANED_INPUT_PATH}"

# 4. Wait for the specific directory to become available on the FUSE mount.
echo "Waiting for source directory to be available at: ${SOURCE_DATA_DIR}"
wait_time=0
max_wait=120 # Wait for a maximum of 2 minutes
while [ ! -d "${SOURCE_DATA_DIR}" ]; do
    if [ $wait_time -ge $max_wait ]; then
        echo "Error: Timed out after ${max_wait}s waiting for ${SOURCE_DATA_DIR}."
        echo "Listing available contents of mount point to debug:"
        ls -l "${MOUNT_POINT}/${PROJECT_NAME}"
        exit 1
    fi
    sleep 5
    wait_time=$((wait_time + 5))
    echo "Still waiting... (${wait_time}s)"
done
echo "Successfully located source directory: ${SOURCE_DATA_DIR}"

# 5. Prepare an argument for the ignore JSON file, if provided.
json_subs2ignore_arg=""
if [ -n "$json_subs2ignore_file" ]; then
    echo "Optional json_subs2ignore_file provided. Creating symbolic link."
    # Describe the file to get its full path within the project
    ignore_file_details=$(dx describe --json "$json_subs2ignore_file")
    ignore_file_folder=$(echo "$ignore_file_details" | jq -r .folder)
    ignore_file_name=$(echo "$ignore_file_details" | jq -r .name)
    
    # Create a symlink to the file from the FUSE mount
    echo "Supplied filename: $json_subs2ignore_file"
    echo "File located at: ${ignore_file_folder}/${ignore_file_name}"
    mkdir -p /home/dnanexus/config
    dx download "$json_subs2ignore_file" -o /home/dnanexus/config/ignore_list.json
    
    # The path inside the container will be /data/in/ignore_list.json
    json_subs2ignore_arg="--json_subs2ignore /data/config/ignore_list.json"
fi

# 6. Create a local directory on the worker for the outputs.
mkdir -p /home/dnanexus/out

# 7. Execute the Docker command.
echo "Executing Docker command..."
docker run --rm \
    --privileged \
    --cap-add SYS_ADMIN \
    --device /dev/fuse \
    -v "${SOURCE_DATA_DIR}":/data/in:ro \
    -v /home/dnanexus/config:/data/config:ro \
    -v /home/dnanexus/out:/data/out \
    "soumickmj/cardiac-diffae-latent-gwas" /bin/bash -c \
    "cd /app/CardiacDiffAE_GWAS/ && python3 preprocess/createH5s/createH5_MR_DICOM.py \
        --copy_zip_locally \
        --in_path /data/in/ \
        --out_path /data/out \
        -dsV \"${dsV}\" \
        --dir_unsorted \
        --fID \"${fID}\" \
        --fDirName \"${fDirName}\" \
        ${json_subs2ignore_arg}"

echo "Docker execution finished."

echo "Listing files in the output directory after execution:"
ls -lR /home/dnanexus/out
cat /home/dnanexus/out/F20208_Long_axis_heart_images_DICOM_H5OnlyV4/log.txt || echo "No log.txt found."

# 8. Upload and Link Outputs
echo "Uploading contents of /home/dnanexus/out as applet outputs..."

# Check if output directory has any content
if [ ! -d "/home/dnanexus/out" ] || [ -z "$(ls -A /home/dnanexus/out 2>/dev/null)" ]; then
    echo "Warning: No output files were found in /home/dnanexus/out to upload."
    # Create an empty JSON array for the output if no files were produced
    OUTPUT_FILE="${DX_JOB_OUTPUT_JSON:-/home/dnanexus/job_output.json}"
    jq -n '{"output_files": []}' > "$OUTPUT_FILE"
else
    # Upload all files and directories recursively as applet outputs
    echo "Uploading files and folders recursively as applet outputs..."
    cd /home/dnanexus/out
    UPLOADED_FILE_IDS=$(dx upload -r * --brief)
    
    # Create proper DNAnexus links from the uploaded file IDs
    echo "Creating output links for uploaded files..."
    if [ -n "$UPLOADED_FILE_IDS" ]; then
        # Convert file IDs to proper DNAnexus link format
        output_ids=$(echo "$UPLOADED_FILE_IDS" | jq -R '{"$dnanexus_link": .}' | jq -s '.')
    else
        output_ids="[]"
    fi
    
    # Set the job output with proper file links
    OUTPUT_FILE="${DX_JOB_OUTPUT_JSON:-/home/dnanexus/job_output.json}"
    jq -n --argjson ids "$output_ids" '{"output_files": $ids}' > "$OUTPUT_FILE"
fi

echo "Applet finished successfully."