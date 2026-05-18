"""
Script to run linear associations between latent factors and diseases/phenotypes.
Processes multiple latent dataframes in a folder and computes associations.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def get_top_diseases(phenos, no_subjects):
    """
    Returns a list of diseases with at least `no_subjects` cases
    in the time window (MRI_Date-5, MRI_Date+1).
    """
    year_condition = (
        (phenos['year'] >= phenos['MRI_Date'] - 5) &
        (phenos['year'] <= phenos['MRI_Date'] + 1)
    )
    filtered_phenos = phenos[year_condition]
    disease_counts = filtered_phenos.groupby('meaning').eid.nunique()
    incident_diseases = disease_counts[disease_counts >= no_subjects].index.tolist()

    print(f"Number of diseases with ≥{no_subjects} subjects: {len(incident_diseases)}")
    return disease_counts, incident_diseases


def preprocess_data_for_disease(phenos, disease_name, latents, covs):
    """
    Prepares dataset for a specific disease:
    - Cases: disease events in time window (MRI_Date-5 to MRI_Date+1).
    - Controls: subjects without the disease.
    - Merges with latent factors and covariates.
    """
    # Identify cases within time window - use exact matching with regex=False
    disease_df = phenos[phenos.meaning.str.contains(disease_name, regex=False)].copy()
    condition = (
        (disease_df['year'] >= disease_df['MRI_Date'] - 5) &
        (disease_df['year'] <= disease_df['MRI_Date'] + 1)
    )
    disease_df[f'recent_{disease_name}'] = condition
    recent_disease = disease_df[disease_df[f'recent_{disease_name}']]
    
    # Get case and control IDs
    case_ids = set(recent_disease.eid.unique())
    all_ids = set(latents.FID.unique())
    control_ids = all_ids - case_ids
    
    # Subset latents
    disease_latents = latents[latents.FID.isin(case_ids)].copy()
    no_disease_latents = latents[latents.FID.isin(control_ids)].copy()

    # Label cases and controls  
    disease_latents['disease'] = 1
    no_disease_latents['disease'] = 0

    # Merge cases + controls with covariates
    df_to_test = pd.concat([disease_latents, no_disease_latents])
    df_to_test = df_to_test.merge(covs, on='FID')
    return df_to_test


def run_regression_loop(df, covariates_list, Z_names):
    """
    Runs OLS regression for each latent separately using statsmodels.
    Returns effect sizes, std errors, and p-values for 'disease'.
    """
    results = []

    X = df[covariates_list + ['disease']]
    X = sm.add_constant(X)

    for latent in Z_names:
        y = df[latent]
        model = sm.OLS(y, X).fit()

        results.append({
            "Latent": latent,
            "Effect Size": model.params['disease'],
            "StdError": model.bse['disease'],
            "p_value": model.pvalues['disease']
        })

    return pd.DataFrame(results)


def linear_associations_fast(
    phenos, latents, covs,
    no_subjects=200,
    covariates_list=['Sex', 'BSA', 'Age', 'MRI_Centre', 'MRI_Date', 'standing_height', 'Waist_circumference']
):
    """
    Main driver function:
    - Finds diseases with enough cases.
    - Preprocesses data for each disease.
    - Runs regression
    - Applies FDR correction.
    """
    results = []

    Z_names = list(latents.columns[2:])  # Skip FID and IID columns

    # Get diseases with ≥ no_subjects
    disease_counts, incident_diseases = get_top_diseases(phenos, no_subjects)

    for disease in incident_diseases:
        print(f"Processing {disease}...")
        df = preprocess_data_for_disease(phenos, disease, latents, covs).dropna()
        
        print(f"Phenotypes tested: {df.shape}")
        
        # print the number of cases and controls
        num_cases = df['disease'].sum()
        num_controls = len(df) - num_cases
        print(f"Number of cases: {num_cases}, Number of controls: {num_controls}")
        print()

        tmp = run_regression_loop(df, covariates_list, Z_names)

        tmp['Disease'] = disease
        results.append(tmp)

    if not results:
        return disease_counts, pd.DataFrame(columns=['Latent', 'Effect Size', 'StdError', 'p_value', 'Disease'])
    
    df_results = pd.concat(results, ignore_index=True)

    # Filter very small effect sizes
    df_results = df_results[
        (df_results['Effect Size'] >= 1e-6) |
        (df_results['Effect Size'] <= -1e-6)
    ].sort_values('p_value')

    return disease_counts, df_results


def process_latent_file(latent_file, covs, phenos, meff, output_dir, 
                        no_subjects=200, covariates_list=None):
    """
    Process a single latent file and compute associations.
    
    Args:
        latent_file: Path to latent TSV file
        covs: Covariate dataframe
        phenos: Disease/phenotype dataframe
        meff: Multiple testing correction factor
        output_dir: Directory to save results
        no_subjects: Minimum number of subjects per disease
        covariates_list: List of covariate column names
    """
    if covariates_list is None:
        covariates_list = ['Sex', 'BSA', 'Age', 'MRI_Centre', 'MRI_Date', 
                          'standing_height', 'Waist_circumference']
    
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(latent_file)}")
    print(f"{'='*80}")
    
    # Load latents
    latents = pd.read_csv(latent_file, delimiter='\t')
    print(f"Loaded latents with shape: {latents.shape}")
    
    # Run associations
    disease_counts, result_df = linear_associations_fast(
        phenos, latents, covs, 
        no_subjects=no_subjects,
        covariates_list=covariates_list
    )
    
    # Get actual number of diseases tested
    n_diseases_tested = result_df['Disease'].nunique() if len(result_df) > 0 else 0
    
    # Add Log10P column
    result_df['Log10P'] = -np.log10(result_df['p_value'])
    result_df = result_df.rename(columns={'p_value': 'P'})
    result_df = result_df[['Disease', 'Latent', 'Effect Size', 'StdError', 'P', 'Log10P']]
    
    # Filter by p-value < 0.05
    nominal_threshold = 0.05
    nominal_sig_df = result_df[result_df['P'] < nominal_threshold].copy()
    n_unique_diseases_nominal = nominal_sig_df['Disease'].nunique()
    diseases_nominal = sorted(nominal_sig_df['Disease'].unique().tolist())
    
    # Filter by adjusted p-value threshold: 0.05 / (disease_count * meff)
    print(f"Applying adjusted p-value threshold with meff={meff}")
    if n_diseases_tested > 0:
        adjusted_threshold = 0.05 / (n_diseases_tested * int(meff))
    else:
        adjusted_threshold = 0.05 / int(meff)
    print(f"Adjusted p-value threshold: {adjusted_threshold:.2e}")
    adjusted_sig_df = result_df[result_df['P'] < adjusted_threshold].copy()
    n_unique_diseases_adjusted = adjusted_sig_df['Disease'].nunique()
    diseases_adjusted = sorted(adjusted_sig_df['Disease'].unique().tolist())
    
    # Save results
    base_name = Path(latent_file).stem
    output_file = os.path.join(output_dir, f"{base_name}_associations.tsv")
    result_df.to_csv(output_file, sep="\t", index=False)
    
    # Save nominal significant results (p < 0.05)
    nominal_output_file = os.path.join(output_dir, f"{base_name}_nominal_sig.tsv")
    nominal_sig_df.to_csv(nominal_output_file, sep="\t", index=False)
    
    # Save adjusted significant results
    adjusted_output_file = os.path.join(output_dir, f"{base_name}_adjusted_sig.tsv")
    adjusted_sig_df.to_csv(adjusted_output_file, sep="\t", index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {base_name}")
    print(f"{'='*80}")
    print(f"Total diseases tested: {n_diseases_tested}")
    print(f"Total associations tested: {len(result_df)}")
    print(f"Nominal threshold (p < 0.05): {nominal_threshold}")
    print(f"  - Significant associations: {len(nominal_sig_df)}")
    print(f"  - Unique diseases: {n_unique_diseases_nominal}")
    print(f"Adjusted threshold (p < 0.05/({n_diseases_tested}*{meff})): {adjusted_threshold:.2e}")
    print(f"  - Significant associations: {len(adjusted_sig_df)}")
    print(f"  - Unique diseases: {n_unique_diseases_adjusted}")
    print(f"Results saved to: {output_file}")
    print(f"Nominal significant results saved to: {nominal_output_file}")
    print(f"Adjusted significant results saved to: {adjusted_output_file}")
    print(f"{'='*80}\n")
    
    return {
        'Latent_Input': base_name,
        'meff': meff,
        'Unique_Diseases_Nominal': n_unique_diseases_nominal,
        'Diseases_Nominal_p<0.05': '; '.join(diseases_nominal) if diseases_nominal else '',
        'Unique_Diseases_Adjusted': n_unique_diseases_adjusted,
        'Diseases_Adjusted': '; '.join(diseases_adjusted) if diseases_adjusted else ''
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run linear associations between latent factors and diseases/phenotypes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python run_linear_associations.py \\
        --latent_folder /path/to/latents/ \\
        --covariates /path/to/covariates.tsv \\
        --phenotypes /path/to/diseases.csv \\
        --meff /path/to/meff_mapping.tsv \\
        --output_dir /path/to/output/
        
The meff file should be a TSV with two columns:
    - First column: filename (without path, matching files in latent_folder)
    - Second column: meff value for that file
        """
    )
    
    parser.add_argument('--latent_folder', required=True,
                       help='Folder containing latent TSV files')
    parser.add_argument('--covariates', required=True,
                       help='Path to covariate TSV file')
    parser.add_argument('--phenotypes', required=True,
                       help='Path to disease/phenotype CSV file')
    parser.add_argument('--meff', required=True,
                       help='Path to TSV file mapping latent filenames to meff values')
    parser.add_argument('--output_dir', required=True,
                       help='Directory to save output files')
    parser.add_argument('--no_subjects', type=int, default=200,
                       help='Minimum number of subjects per disease (default: 200)')
    parser.add_argument('--covariates_list', nargs='+',
                       default=['Sex', 'BSA', 'Age', 'MRI_Centre', 'MRI_Date', 
                               'standing_height', 'Waist_circumference'],
                       help='List of covariate column names')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.latent_folder):
        print(f"Error: Latent folder does not exist: {args.latent_folder}")
        sys.exit(1)
    
    if not os.path.exists(args.covariates):
        print(f"Error: Covariate file does not exist: {args.covariates}")
        sys.exit(1)
    
    if not os.path.exists(args.phenotypes):
        print(f"Error: Phenotype file does not exist: {args.phenotypes}")
        sys.exit(1)
    
    if not os.path.exists(args.meff):
        print(f"Error: Meff mapping file does not exist: {args.meff}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load meff mapping
    print("Loading meff mapping...")
    meff_df = pd.read_csv(args.meff, sep='\t', names=['filename', 'meff'])
    meff_dict = dict(zip(meff_df['meff'], meff_df['filename']))
    print(meff_dict)
    print(f"Loaded meff values for {len(meff_dict)} files")
    
    # Load covariates and phenotypes
    print("Loading covariates...")
    covs = pd.read_csv(args.covariates, sep='\t')
    print(f"Loaded covariates with shape: {covs.shape}")
    
    print("Loading phenotypes...")
    phenos = pd.read_csv(args.phenotypes).dropna()
    if 'year' in phenos.columns:
        phenos['year'] = phenos['year'].astype(int)
    print(f"Loaded phenotypes with shape: {phenos.shape}")
    
    # Find all TSV files in latent folder
    latent_files = sorted([
        os.path.join(args.latent_folder, f)
        for f in os.listdir(args.latent_folder)
        if f.endswith('.tsv')
    ])
    
    if not latent_files:
        print(f"Error: No TSV files found in {args.latent_folder}")
        sys.exit(1)
    
    print(f"\nFound {len(latent_files)} latent file(s) to process")
    
    # Process each latent file
    summary_results = []
    for latent_file in latent_files:
        filename = os.path.basename(latent_file)
        print(f"\nProcessing file: {filename}")

        # Get meff for this file
        if filename not in meff_dict:
            print(f"\nWarning: No meff value found for {filename}, skipping...")
            continue
        
        file_meff = meff_dict[filename]
        print(f"\nUsing meff={file_meff} for {filename}")
        
        result = process_latent_file(
            latent_file, covs, phenos, file_meff, args.output_dir,
            no_subjects=args.no_subjects,
            covariates_list=args.covariates_list
        )
        summary_results.append(result)
    
    # Save overall summary
    summary_df = pd.DataFrame(summary_results)
    summary_file = os.path.join(args.output_dir, 'overall_summary.tsv')
    summary_df.to_csv(summary_file, sep='\t', index=False)
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_file}")
    print("="*80)


if __name__ == '__main__':
    main()
