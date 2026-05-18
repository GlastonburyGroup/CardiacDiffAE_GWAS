# %%
import pandas as pd
import numpy as np  
from glob import glob
from tqdm import tqdm
from scipy.stats import combine_pvalues, chi2, norm
from scipy.linalg import cholesky
import argparse
import logging
import os
from pathlib import Path

def calculate_maf(a1freq):
    a2freq = 1 - a1freq
    return min(a1freq, a2freq)

def filter_HLA(df, col_chrom='CHROM', col_genpos='GENPOS'):
    hla_start = 28_477_797
    hla_end = 33_448_354
    chromosome_of_hla = 6

    return df[~((df[col_chrom] == chromosome_of_hla) & 
                (df[col_genpos] >= hla_start) & 
                (df[col_genpos] <= hla_end))]

def combine_pvals(pvals, method='fisher'):
    return combine_pvalues(pvals, method=method)[1]

def simes_method(pvals):
    pvals = np.sort(pvals)
    n = len(pvals)
    simes_p = np.min(pvals * n / np.arange(1, n + 1))
    return min(simes_p, 1.0)

def empirical_brown_method(pvals, corr_matrix):
    n = len(pvals)
    z_scores = norm.ppf(1 - pvals)

    L = cholesky(corr_matrix, lower=True)
    
    new_z = np.dot(L, z_scores)
    new_chi2 = np.sum(new_z**2)
    
    e_df = np.trace(corr_matrix)
    
    combined_p = 1 - norm.cdf(np.sqrt(new_chi2 / e_df))
    return combined_p

def calculate_corr_matrix(df, groupby_cols, pval_col):
    grouped = df.groupby(groupby_cols)
    pval_matrix = grouped[pval_col].apply(np.array).unstack()
    pval_matrix = pval_matrix.fillna(pval_matrix.mean())
    pval_matrix = pval_matrix.apply(norm.ppf, args=(1,))
    corr_matrix = np.corrcoef(pval_matrix, rowvar=False)
    return corr_matrix

def lancaster_method(pvals, weights=None):
    n = len(pvals)
    chi2_stats = chi2.isf(pvals, df=1)
    if weights is None:
        weights = np.ones(n)
    combined_chi2 = np.sum(weights * chi2_stats)
    combined_p = chi2.sf(combined_chi2, df=2*n)  
    return combined_p

def inverse_variance_weighted_method(betas, ses):
    weights = 1 / (ses ** 2)
    combined_beta = np.sum(weights * betas) / np.sum(weights)
    combined_se = np.sqrt(1 / np.sum(weights))
    z_score = combined_beta / combined_se
    combined_p = 2 * (1 - norm.cdf(abs(z_score)))  
    return combined_p

def harmonic_mean_pvalue(pvals):
    n = len(pvals)
    harmonic_mean = n / np.sum(1.0 / pvals)
    combined_p = min(1, harmonic_mean)
    return combined_p

def setup_logging(verbose=False):
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge GWAS results using various p-value combination methods.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-pattern',
        type=str,
        required=True,
        help='Glob pattern for input GWAS files (e.g., "/path/to/*.gwas.regenie.gz")'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save output files'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1_000_000,
        help='Number of rows to process per chunk'
    )
    
    parser.add_argument(
        '--maf-threshold',
        type=float,
        default=0.01,
        help='Minimum minor allele frequency threshold'
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['minP', 'fisher', 'stouffer', 'lancaster', 'simes', 'harmonic', 'ivw', 'ebm'],
        default=['minP'],
        help='P-value combination methods to use'
    )
    
    parser.add_argument(
        '--filter-hla',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Filter out HLA region (chr6:28477797-33448354)'
    )
    
    parser.add_argument(
        '--verbose',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable verbose logging'
    )
    
    # Column name mappings
    parser.add_argument(
        '--col-id',
        type=str,
        default='ID',
        help='Column name for variant ID'
    )
    
    parser.add_argument(
        '--col-chrom',
        type=str,
        default='CHROM',
        help='Column name for chromosome'
    )
    
    parser.add_argument(
        '--col-genpos',
        type=str,
        default='GENPOS',
        help='Column name for genomic position'
    )
    
    parser.add_argument(
        '--col-a1freq',
        type=str,
        default='A1FREQ',
        help='Column name for allele 1 frequency'
    )
    
    parser.add_argument(
        '--col-log10p',
        type=str,
        default='LOG10P',
        help='Column name for -log10(p-value)'
    )
    
    parser.add_argument(
        '--col-pval',
        type=str,
        default=None,
        help='Column name for p-value (use this if you have p-values instead of -log10(p-values)). If provided, LOG10P will be calculated from this column.'
    )
    
    parser.add_argument(
        '--col-beta',
        type=str,
        default='BETA',
        help='Column name for effect size/beta (required for IVW method)'
    )
    
    parser.add_argument(
        '--col-se',
        type=str,
        default='SE',
        help='Column name for standard error (required for IVW method)'
    )
    
    return parser.parse_args()

def process_methods(df, minP, methods, col_id='ID', col_p='P', col_beta='BETA', col_se='SE'):
    """Process selected p-value combination methods and return results."""
    results = {}
    
    if 'fisher' in methods:
        try:
            logging.info("Calculating Fisher's combined p-values...")
            fisher = df.groupby(col_id)[col_p].apply(lambda x: combine_pvals(x.values, method='fisher'))
            fisher_df = fisher.reset_index()
            fisher_df.rename(columns={col_p: 'fisherP'}, inplace=True)
            fisherP = pd.merge(minP, fisher_df, on=[col_id], how='left')
            results['fisherP'] = fisherP
        except Exception as e:
            logging.error(f"Error in fisherP: {e}")

    if 'stouffer' in methods:
        try:
            logging.info("Calculating Stouffer's combined p-values...")
            stouffer = df.groupby(col_id)[col_p].apply(lambda x: combine_pvals(x.values, method='stouffer'))
            stouffer_df = stouffer.reset_index()
            stouffer_df.rename(columns={col_p: 'stoufferP'}, inplace=True)
            stoufferP = pd.merge(minP, stouffer_df, on=[col_id], how='left')
            results['stoufferP'] = stoufferP
        except Exception as e:
            logging.error(f"Error in stoufferP: {e}")

    if 'lancaster' in methods:
        try:
            logging.info("Calculating Lancaster combined p-values...")
            lancaster = df.groupby(col_id)[col_p].apply(lambda x: lancaster_method(x.values))
            lancaster_df = lancaster.reset_index()
            lancaster_df.rename(columns={col_p: 'lancasterP'}, inplace=True)
            lancasterP = pd.merge(minP, lancaster_df, on=[col_id], how='left')
            results['lancasterP'] = lancasterP
        except Exception as e:
            logging.error(f"Error in lancasterP: {e}")

    if 'simes' in methods:
        try:
            logging.info("Calculating Simes combined p-values...")
            simes = df.groupby(col_id)[col_p].apply(lambda x: simes_method(x.values))
            simes_df = simes.reset_index()
            simes_df.rename(columns={col_p: 'simesP'}, inplace=True)
            simesP = pd.merge(minP, simes_df, on=[col_id], how='left')
            results['simesP'] = simesP
        except Exception as e:
            logging.error(f"Error in simesP: {e}")

    if 'harmonic' in methods:
        try:
            logging.info("Calculating harmonic mean p-values...")
            harmonicMean = df.groupby(col_id)[col_p].apply(lambda x: harmonic_mean_pvalue(x.values))
            harmonicMean_df = harmonicMean.reset_index()
            harmonicMean_df.rename(columns={col_p: 'harmonicMeanP'}, inplace=True)
            harmonicMeanP = pd.merge(minP, harmonicMean_df, on=[col_id], how='left')
            results['harmonicMeanP'] = harmonicMeanP
        except Exception as e:
            logging.error(f"Error in harmonicMeanP: {e}")

    if 'ivw' in methods:
        try:
            logging.info("Calculating inverse variance weighted p-values...")
            combined_pvalues = []
            for id_, group in df.groupby([col_id]):
                betas = group[col_beta].values
                ses = group[col_se].values
                combined_p = inverse_variance_weighted_method(betas, ses)
                combined_pvalues.append({col_id: id_, 'ivwP': combined_p})
            combined_pvalues_df = pd.DataFrame(combined_pvalues)
            ivwP = pd.merge(minP, combined_pvalues_df, on=[col_id], how='left')
            results['ivwP'] = ivwP
        except Exception as e:
            logging.error(f"Error in ivwP: {e}")

    if 'ebm' in methods:
        try:
            logging.info("Calculating empirical Brown method p-values...")
            combined_pvalues = []
            for id_, group in df.groupby(col_id):
                pvals = group[col_p].values
                corr_matrix = calculate_corr_matrix(df, col_id, col_p)
                combined_p = empirical_brown_method(pvals, corr_matrix)
                combined_pvalues.append({col_id: id_, 'ebmP': combined_p})
            combined_pvalues_df = pd.DataFrame(combined_pvalues)
            ebmP = pd.merge(minP, combined_pvalues_df, on=[col_id], how='left')
            results['ebmP'] = ebmP
        except Exception as e:
            logging.error(f"Error in ebmP: {e}")
    
    return results

# %%
def main():
    """Main execution function."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    # Validate column configuration
    if args.col_pval and args.col_log10p == 'LOG10P':
        logging.info(f"Using p-value column '{args.col_pval}' - will calculate -log10(p)")
    elif args.col_pval:
        logging.warning(f"Both --col-pval ('{args.col_pval}') and --col-log10p ('{args.col_log10p}') provided. Using --col-pval and ignoring --col-log10p.")
    
    # Validate inputs
    input_files = glob(args.input_pattern.strip('"').strip("'"))
    if not input_files:
        logging.error(f"No files found matching pattern: {args.input_pattern}")
        return
    
    logging.info(f"Found {len(input_files)} input files")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Determine total rows and chunks
    logging.info("Reading first file to determine chunk size...")
    df = pd.read_csv(input_files[0], sep='\t', compression='gzip')
    total_rows = df.shape[0]
    n_chunks = total_rows // args.chunk_size + 1
    del df
    
    logging.info(f"Total rows: {total_rows}, Processing in {n_chunks} chunks of {args.chunk_size} rows")
    
    # Initialise collectors for each method
    collectors = {method: [] for method in args.methods}
    
    # Process chunks
    for i in tqdm(range(n_chunks), desc="Processing chunks"):
        start_idx = i * args.chunk_size
        nrows = (total_rows - start_idx) if i + 1 == n_chunks else args.chunk_size
        
        # Load and concatenate data from all input files
        df_collect = []
        for file_path in input_files:
            df = pd.read_csv(file_path, compression='gzip', sep='\t', 
                           skiprows=range(1, start_idx + 1), nrows=nrows)
            df['MAF'] = df[args.col_a1freq].apply(calculate_maf)    
            df = df[df.MAF > args.maf_threshold]
            
            if args.filter_hla:
                df = filter_HLA(df, col_chrom=args.col_chrom, col_genpos=args.col_genpos)
            
            df["phenotype"] = Path(file_path).stem.split(".")[0]
            df_collect.append(df)
        
        df = pd.concat(df_collect)
        
        # Calculate P-value from either p-value column or -log10(p) column
        if args.col_pval:
            df["P"] = df[args.col_pval]
        else:
            df["P"] = 10 ** -df[args.col_log10p]
        
        # Always calculate minP as base
        minP = df.sort_values("P").drop_duplicates(args.col_id, keep="first").sort_values([args.col_chrom, args.col_genpos])
        if 'minP' in args.methods:
            collectors['minP'].append(minP)
        
        # Process other methods
        chunk_results = process_methods(df, minP, args.methods, 
                                       col_id=args.col_id, 
                                       col_p='P',
                                       col_beta=args.col_beta, 
                                       col_se=args.col_se)
        for method_name, result_df in chunk_results.items():
            collectors[method_name].append(result_df)
    
    # Save results
    logging.info("Saving results...")
    for method_name, data_list in collectors.items():
        if len(data_list) > 0:
            output_file = output_dir / f"merge_attempts_{method_name}.tsv"
            logging.info(f"Writing {method_name} results to {output_file}")
            combined_df = pd.concat(data_list)
            combined_df.to_csv(output_file, sep="\t", index=False)
            del combined_df
    
    logging.info("Processing complete!")

if __name__ == "__main__":
    main()
