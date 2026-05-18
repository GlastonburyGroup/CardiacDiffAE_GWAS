import pandas as pd
import numpy as np
import glob
import os
import sys
import statsmodels.stats.multitest as smm

def rank_latent_models(output_dir, fdr_threshold=0.05):
    """
    Ranks latent association files by calculating FDR from scratch
    and comparing statistical strength.
    """
    # Find all main association files
    files = glob.glob(os.path.join(output_dir, "*_associations.tsv"))
    
    if not files:
        print(f"No association files found in {output_dir}")
        return

    print(f"Analysing {len(files)} models. Recalculating FDR (Benjamini-Hochberg)...\n")
    
    ranking_data = []

    for f in files:
        df = pd.read_csv(f, sep='\t')
        filename = os.path.basename(f).replace('_associations.tsv', '')
        
        # ---------------------------------------------------------
        # 1. Recalculate FDR (Benjamini-Hochberg)
        # ---------------------------------------------------------
        # We use the raw 'P' column. 
        # method='fdr_bh' ensures we control the False Discovery Rate.
        if not df.empty:
            reject, pvals_corrected, _, _ = smm.multipletests(
                df['P'], 
                alpha=fdr_threshold, 
                method='fdr_bh'
            )
            n_significant_fdr = np.sum(reject)
        else:
            n_significant_fdr = 0

        # ---------------------------------------------------------
        # 2. Calculate Wald Statistic (t^2)
        # ---------------------------------------------------------
        # t = Effect Size / StdError
        # We square it to measure magnitude of signal vs noise
        with np.errstate(divide='ignore', invalid='ignore'):
            df['t_stat'] = df['Effect Size'] / df['StdError']
        
        # Clean up infinites/NaNs for aggregation
        clean_t = df['t_stat'].replace([np.inf, -np.inf], np.nan).dropna()
        t_squared = clean_t ** 2

        # Metric: Mean Signal Strength
        mean_t2 = t_squared.mean()

        # ---------------------------------------------------------
        # 3. Calculate Inflation (Lambda)
        # ---------------------------------------------------------
        # Median of Chi-squared(1) is approx 0.4549364
        median_t2 = t_squared.median()
        lambda_val = median_t2 / 0.4549364

        # ---------------------------------------------------------
        # 4. Precision (Median SE)
        # ---------------------------------------------------------
        median_se = df['StdError'].median()

        ranking_data.append({
            'Model': filename,
            'FDR_Significant_Hits': n_significant_fdr,
            'Mean_Chi2_Strength': mean_t2,
            'Inflation_Lambda': lambda_val,
            'Median_SE': median_se
        })

    # ---------------------------------------------------------
    # Generate Rankings
    # ---------------------------------------------------------
    rank_df = pd.DataFrame(ranking_data)

    # Sort Logic:
    # 1. Primary: Count of FDR Hits (Descending)
    # 2. Secondary: Average Signal Strength (Descending)
    # 3. Tertiary: Precision/SE (Ascending - lower error is better)
    rank_df = rank_df.sort_values(
        by=['FDR_Significant_Hits', 'Mean_Chi2_Strength', 'Median_SE'], 
        ascending=[False, False, True]
    ).reset_index(drop=True)

    # Add Rank Number
    rank_df.index += 1
    rank_df.index.name = 'Rank'

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    print("="*110)
    print(f"STATISTICAL MODEL RANKING (FDR Threshold < {fdr_threshold})")
    print("="*110)
    
    # Format columns for readability
    output_df = rank_df.copy()
    output_df['Mean_Chi2_Strength'] = output_df['Mean_Chi2_Strength'].map('{:.4f}'.format)
    output_df['Inflation_Lambda'] = output_df['Inflation_Lambda'].map('{:.4f}'.format)
    output_df['Median_SE'] = output_df['Median_SE'].map('{:.6f}'.format)
    
    print(output_df.to_string())
    print("="*110)
    print("\nStatistical Definitions:")
    print(f"1. FDR_Significant_Hits: Number of associations with Benjamini-Hochberg adj. p-value < {fdr_threshold}.")
    print("2. Mean_Chi2_Strength: Average (Effect/SE)^2. Indicates total explanatory power of the latents.")
    print("3. Inflation_Lambda: Median(t^2)/0.455. Values > 1 indicate the latents contain systematic signal.")
    print("4. Median_SE: Median Standard Error. Lower values indicate more stable/precise predictions.")

rank_latent_models("/group/glastonbury/soumick/rough/dis_assoc_liv_Emma/")