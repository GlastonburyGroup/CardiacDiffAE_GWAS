import pandas as pd

def adjust_merged_loci(all_loci, loci_radius=250000):
    # Sort data by chromosome and base pair position
    all_loci.sort_values(['CHR', 'BP'], inplace=True)

    # Initialise an empty DataFrame for the final loci
    final_loci = pd.DataFrame()

    # Group by chromosome
    for chrom, chrom_data in all_loci.groupby('CHR'):
        # Initialise the first locus with the first SNP
        current_locus_centre = chrom_data.iloc[0]['BP']
        current_locus_data = [chrom_data.iloc[0]]

        # Process the remaining SNPs in the chromosome
        for _, row in chrom_data.iloc[1:].iterrows():
            # Check if the SNP falls within the current locus
            if abs(row['BP'] - current_locus_centre) <= loci_radius:
                # If it does, add it to the current locus data
                current_locus_data.append(row)
            else:
                # If it doesn't, finalise the current locus and start a new one
                final_loci = finalise_locus(final_loci, current_locus_data)

                # Start a new locus
                current_locus_centre = row['BP']
                current_locus_data = [row]

        # Finalise the last locus of the chromosome
        final_loci = finalise_locus(final_loci, current_locus_data)

    return final_loci


def finalise_locus(final_loci, current_locus_data):
    current_locus_df = pd.DataFrame(current_locus_data)
    current_locus_df = current_locus_df.reset_index(drop=True)
    lead_snp = current_locus_df.loc[current_locus_df['P'].idxmin()].copy()  # The SNP with the smallest p-value

    # Add the other SNPs in the locus to SP2 of the lead SNP
    other_lead_snps = current_locus_df[current_locus_df['SNP'] != lead_snp['SNP']]
    if not other_lead_snps.empty:
        if lead_snp['SP2'] == "NONE":
            lead_snp['SP2'] = ','.join(other_lead_snps['SNP']) + ',' + ','.join(other_lead_snps['SP2'])
        else:
            lead_snp['SP2'] += ',' + ','.join(other_lead_snps['SNP']) + ',' + ','.join(other_lead_snps['SP2'])

    # Sum up the relevant columns
    for col in ['TOTAL', 'NSIG', 'S05', 'S01', 'S001', 'S0001']:
        lead_snp[col] = current_locus_df[col].sum()

    # Concatenate the phenotypes, but only include unique values
    lead_snp['Pheno'] = ','.join(current_locus_df['Pheno'].unique())

    if 'Run' in lead_snp: #For multi-Run comparison, this will be utilised
        lead_snp['Run'] = ','.join(current_locus_df['Run'].unique())
        lead_snp['RunPheno'] = ','.join((current_locus_df['Run'] + "§" + current_locus_df['Pheno']).unique())

    # Append the lead SNP of the locus to the final DataFrame
    # final_loci = final_loci.append(lead_snp, ignore_index=True)
    final_loci = pd.concat([final_loci, lead_snp.to_frame().transpose()], ignore_index=True)
    
    return final_loci