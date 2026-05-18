# UK Biobank Image Processing

## Overview

This guide provides step-by-step instructions for creating and processing the UK Biobank imaging dataset (e.g. 20208 Long axis heart images - DICOM). The dataset consists of MRI scans in DICOM format, packaged as ZIP files, which need to be converted into HDF5 format for further analysis.

## Dataset Information

- **Data Format**: ZIP files containing DICOM series
- **Output Format**: HDF5 (.h5) files with organised cardiac imaging data

## Prerequisites

### For Local/Python Execution:
- Python 3.8+
- Required packages (install via pip or conda):
  - `h5py`
  - `pandas`
  - `numpy`
  - `pydicom` or `SimpleITK`
  - `pyyaml`
  - `tqdm`
  - `tricorder` (for DICOM reading utilities)

### For Docker Execution:
- Docker installed and running
- Access to the Docker image: `soumickmj/cardiac-diffae-latent-gwas` (please note this Docker image was created for field ID 20208, but should be fine for others too)

### For UK Biobank RAP (DNAnexus) Execution:
- Access to UK Biobank Research Analysis Platform (RAP)
- DNAnexus CLI tools installed (`dx` command)
- Project access permissions

---

## Step 1: Create HDF5 Dataset from ZIP Files

This step converts the raw DICOM ZIP files into organised HDF5 files for efficient data access and processing.

### Method 1A: Using Python Script Locally

#### Script Location
```
preprocess/createH5s/createH5_MR_DICOM.py
```

#### Basic Usage
```bash
python preprocess/createH5s/createH5_MR_DICOM.py \
    --in_path /path/to/ukb/imaging/zipfiles \
    --out_path /path/to/output/dataset \
    -dsV "v1" \
    --dir_unsorted \
    --fID "20208" \
    --fDirName "F20208_Long_axis_heart_images_DICOM"
```

#### Command-Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--in_path` | Path to the directory containing ZIP files | `../imaging` | No |
| `--out_path` | Path to store the HDF5 output files | `../dataset/ukbbH5s` | No |
| `--use_SimpleITK` | Use SimpleITK instead of PyDicom for reading | `True` | No |
| `-dsV` | Dataset version string (appended to output directory) | `"2"` | No |
| `--dir_unsorted` | Process unsorted bulk files (freshly downloaded) | `False` | No |
| `--fID` | Field ID of the bulk files to process | `""` | Yes (if `--dir_unsorted`) |
| `--fDirName` | Name of the directory for the particular field | `""` | Yes (if `--dir_unsorted`) |
| `--add_unsorted` | Additionally process unsorted directory | `""` | No |
| `--json_subs2ignore` | JSON file with list of subject IDs to skip | `""` | No |
| `--copy_zip_locally` | Copy ZIP files to temp before unzipping | `False` | No |

#### Example: Processing Field 20208
```bash
python preprocess/createH5s/createH5_MR_DICOM.py \
    --in_path /data/ukbb/bulk/20208/ \
    --out_path /data/ukbb/processed/ \
    -dsV "v3" \
    --dir_unsorted \
    --fID "20208" \
    --fDirName "F20208_Long_axis_heart_images_DICOM" \
    --json_subs2ignore /data/ukbb/already_processed_subjects.json \
    --copy_zip_locally
```

#### Output Structure
The script creates:
```
/data/ukbb/processed/F20208_Long_axis_heart_images_DICOM_H5v3/
├── data.h5          # Main HDF5 file with all subjects
└── log.txt          # Processing log with warnings and errors
```

#### HDF5 File Structure
```
data.h5
└── <patientID>/
    └── <fieldID>/
        └── <instanceID>/
            ├── primary              # Main cardiac CINE data
            ├── primary_<plane>      # Data from different orientations
            ├── auxiliary_<tag>      # Auxiliary data (if present)
            └── [attributes]         # Metadata (study info, series info, etc.)
```

---

### Method 1B: Using Docker Container

#### Docker Image
```
soumickmj/cardiac-diffae-latent-gwas
```

#### Basic Docker Command
```bash
docker run --rm \
    -v /path/to/input/zipfiles:/data/in:ro \
    -v /path/to/output:/data/out \
    soumickmj/cardiac-diffae-latent-gwas /bin/bash -c \
    "cd /app/CardiacDiffAE_GWAS/ && python3 preprocess/createH5s/createH5_MR_DICOM.py \
        --in_path /data/in/ \
        --out_path /data/out \
        -dsV 'v1' \
        --dir_unsorted \
        --fID '20208' \
        --fDirName 'F20208_Long_axis_heart_images_DICOM'"
```

#### With Optional Ignore List
```bash
docker run --rm \
    -v /path/to/input/zipfiles:/data/in:ro \
    -v /path/to/config:/data/config:ro \
    -v /path/to/output:/data/out \
    soumickmj/cardiac-diffae-latent-gwas /bin/bash -c \
    "cd /app/CardiacDiffAE_GWAS/ && python3 preprocess/createH5s/createH5_MR_DICOM.py \
        --in_path /data/in/ \
        --out_path /data/out \
        -dsV 'v1' \
        --dir_unsorted \
        --fID '20208' \
        --fDirName 'F20208_Long_axis_heart_images_DICOM' \
        --json_subs2ignore /data/config/ignore_list.json"
```

#### Docker with FUSE Support (for Cloud/Network Storage)
```bash
docker run --rm \
    --privileged \
    --cap-add SYS_ADMIN \
    --device /dev/fuse \
    -v /path/to/input:/data/in:ro \
    -v /path/to/output:/data/out \
    soumickmj/cardiac-diffae-latent-gwas /bin/bash -c \
    "cd /app/CardiacDiffAE_GWAS/ && python3 preprocess/createH5s/createH5_MR_DICOM.py \
        --copy_zip_locally \
        --in_path /data/in/ \
        --out_path /data/out \
        -dsV 'v1' \
        --dir_unsorted \
        --fID '20208' \
        --fDirName 'F20208_Long_axis_heart_images_DICOM'"
```

**Note**: Use `--copy_zip_locally` flag when reading from network-mounted filesystems (like FUSE mounts) to avoid streaming issues.

---

### Method 1C: Using UK Biobank RAP (DNAnexus) Applet

#### Applet Location
```
DNANexus_UKBRAP/applets/create_h5_applet/
```

#### Step 1C.1: Build and Upload the Applet

1. **Navigate to the applet directory:**
   ```bash
   cd DNANexus_UKBRAP/applets/create_h5_applet/
   ```

2. **Ensure you're logged into DNAnexus:**
   ```bash
   dx login
   dx select <your-project-name>
   ```

3. **Build and upload the applet:**
   ```bash
   dx build -f create_h5_applet
   ```

   This will compile the applet and upload it to your current DNAnexus project.

#### Step 1C.2: Run the Applet via Web Interface

1. **Navigate to your project** on the UK Biobank RAP web interface
2. **Go to Tools** → Find `create_h5_applet`
3. **Click "Run"** and configure the inputs:

   | Input Parameter | Example Value | Description |
   |----------------|---------------|-------------|
   | `in_path` | `/Bulk/Heart MRI/Long axis/` | Path to DICOM ZIP files |
   | `json_subs2ignore_file` | (optional) | File with subject IDs to skip |
   | `dsV` | `"v4"` or `"OnlyV4"` | Dataset version identifier |
   | `fID` | `"20208"` | Field ID |
   | `fDirName` | `"F20208_Long_axis_heart_images_DICOM"` | Output directory name |

4. **Click "Start Analysis"**

#### Step 1C.3: Run the Applet via CLI

```bash
dx run create_h5_applet \
    -iin_path="/Bulk/Heart MRI/Long axis/" \
    -idsV="v4" \
    -ifID="20208" \
    -ifDirName="F20208_Long_axis_heart_images_DICOM" \
    --destination=/output/h5_datasets/ \
    --instance-type=mem2_hdd2_v2_x2 \
    --yes
```

#### With Optional Ignore List:
```bash
dx run create_h5_applet \
    -iin_path="/Bulk/Heart MRI/Long axis/" \
    -ijson_subs2ignore_file=file-xxxxxxxxxxxx \
    -idsV="v4" \
    -ifID="20208" \
    -ifDirName="F20208_Long_axis_heart_images_DICOM" \
    --destination=/output/h5_datasets/ \
    --instance-type=mem2_hdd2_v2_x2 \
    --yes
```

**Note**: Replace `file-xxxxxxxxxxxx` with the actual DNAnexus file ID of your ignore list JSON file.

#### Step 1C.4: Monitor Job Progress

```bash
dx watch <job-id>
```

Or monitor via the web interface under **Monitor** → **Jobs**.

#### Applet Configuration

The applet is configured in `dxapp.json` with:
- **Timeout**: 48 hours
- **Instance Type**: `mem2_hdd2_v2_x2` (AWS eu-west-2)
- **Distribution**: Ubuntu 20.04
- **Privileges**: Requires privileged execution for FUSE mounting

---

## Understanding the Configuration File

The processing behaviour is controlled by `preprocess/createH5s/meta.yaml`. For Field 20208, ensure the configuration includes:

```yaml
F20208:
  multi_channel: false
  is_dynamic: true      # CINE sequences are dynamic (time series)
  is_3D: false          # 2D slices
  is_complex: true      # Magnitude and phase data
  repeat_acq: true      # Multiple acquisitions may exist
  default_plane: "transverse"
  desctags:
    primary_data:
      - ["cine_long_axis", "LAX CINE"]
    primary_data_tags:
      - "2ch"
      - "3ch"
      - "4ch"
    auxiliary_data: []
    auxiliary_data_tags: []
```

---

## Output Verification

After processing, verify the output:

### Check Log File
```bash
cat /path/to/output/F20208_Long_axis_heart_images_DICOM_H5v*/log.txt
```

Look for:
- Warnings about missing or corrupted DICOM files
- Errors in specific ZIP files
- Summary of processed subjects

### Inspect HDF5 File
```python
import h5py

with h5py.File('/path/to/output/F20208_Long_axis_heart_images_DICOM_H5v1/data.h5', 'r') as f:
    # List all subject IDs
    print(f"Total subjects: {len(f.keys())}")
    
    # Inspect a specific subject
    subject_id = list(f.keys())[0]
    print(f"\nSubject: {subject_id}")
    print(f"Fields: {list(f[subject_id].keys())}")
    
    # Check instance data
    instance_path = f"{subject_id}/20208"
    if instance_path in f:
        instance = f[instance_path]
        for inst_id in instance.keys():
            print(f"\nInstance: {inst_id}")
            print(f"Datasets: {list(instance[inst_id].keys())}")
            
            # Check primary data shape
            if 'primary' in instance[inst_id]:
                data_shape = instance[inst_id]['primary'].shape
                print(f"Primary data shape: {data_shape}")
                print(f"Attributes: {dict(instance[inst_id]['primary'].attrs)}")
```

### Expected Data Dimensions
For Field 20208 (CINE long axis):
- **Shape**: `(channels, time_frames, slices, height, width)`
- **Typical**: `(1, 50, 1, 256, 256)` or similar
- **Complex data**: May have both magnitude and phase components

---

## Common Issues and Troubleshooting

### Issue 1: "Dirty DICOM" Errors
**Symptom**: Log shows "Dirty DICOM" errors for certain series.
**Solution**: These are automatically skipped. Review the log to see which subjects/series were affected.

### Issue 2: Memory Issues
**Symptom**: Process crashes or runs out of memory.
**Solution**: 
- For local execution: Use a machine with more RAM (16GB+ recommended)
- For Docker: Increase Docker's memory allocation
- For RAP: Use a larger instance type (e.g., `mem3_ssd1_v2_x8`)

### Issue 3: FUSE Mount Failures (RAP)
**Symptom**: "Timed out waiting for source directory"
**Solution**: The applet waits up to 2 minutes. If still failing:
- Check the input path is correct
- Ensure you have read permissions
- Verify the data exists in the specified location

### Issue 4: Complex Data Mismatched
**Symptom**: Error about magnitude/phase acquisition numbers not matching.
**Solution**: This indicates an issue with the source DICOM data. The problematic series will be skipped.

### Issue 5: Missing Dependencies
**Symptom**: Import errors for `tricorder` or other packages.
**Solution**: 
```bash
pip install tricorder pydicom h5py pandas numpy pyyaml tqdm
```

---

## Next Steps

After successfully creating the HDF5 dataset, the typical workflow continues with:

1. **Quality Control**: Inspect the data for completeness and quality
2. **Preprocessing**: Further image processing (normalisation, cropping, etc.)
3. **Feature Extraction**: Extract latent features or perform segmentation
4. **Analysis**: GWAS or other downstream analyses

*(Instructions for these steps will be provided in separate documents)*

---

## Summary of Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Python Script** | Local processing, small datasets | Direct control, easy debugging | Requires local storage and compute |
| **Docker** | Reproducibility, various environments | Consistent environment, portable | Requires Docker setup |
| **RAP Applet** | Large-scale UKB processing | Scalable, cloud-native, no data transfer | Requires UKB RAP access, more complex setup |

---

## Additional Resources

- **UK Biobank RAP Documentation**: https://dnanexus.gitbook.io/uk-biobank-rap/
- **DNAnexus CLI Guide**: https://documentation.dnanexus.com/user/helpstrings-of-sdk-command-line-utilities
- **Field 20208 Showcase**: https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20208

---

## Support

For issues or questions:
1. Check the `log.txt` file in the output directory
2. Review the troubleshooting section above
3. Contact the repository maintainers with specific error messages

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Repository**: CardiacDiffAE_GWAS
