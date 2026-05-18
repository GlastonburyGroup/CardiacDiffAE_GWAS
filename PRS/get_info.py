import numpy as np
from pybgen import PyBGEN
from pybgen.parallel import ParallelPyBGEN
from tqdm import tqdm

def get_info_score(variant):
    """
    Calculate the INFO score for a single SNP based on genotype dosages.
    
    Parameters:
    - genotypes: A numpy array of imputed genotype dosages for individuals (values between 0 and 2).
    
    Returns:
    - INFO score as a float.
    """
    p = np.mean(variant) / 2
    q = 1 - p
    var_g = np.var(variant, ddof=1)  # ddof=1 for sample variance
    
    if p > 0 and q > 0:  # Prevent division by zero
        info_score = var_g / (2 * p * q)
    else:
        info_score = np.nan  # Assign NaN if p or q is 0
    
    return info_score

if __name__ == '__main__':
    bgen_path = '/scratch/edoardo.giacopuzzi/UKBB/step2_dataset/step2_dataset_autosomes.mac100.bgen'
    output_path = '/group/glastonbury/soumick/PRS/inputs/common/SNPs_infGT0p4_ukbb_autosomes_mac100.txt'

    with open(output_path, 'w') as output_file:
        with ParallelPyBGEN(bgen_path, cpus=32, max_variants=100000) as bgen:  
            for variant in tqdm(bgen, total=bgen.nb_variants):
                info_score = get_info_score(variant[1])  
                if info_score > 0.4:
                    output_file.write(f"{variant[0].name.split('_')[0]}\n")  
