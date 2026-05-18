import requests
import json
import pandas as pd
from io import StringIO
import time
import random
import logging

TOKENS = ["63647a5b70ba", "20d5940fb7a8", "9de42c5a2ac3", "dcaa26c0bdd9", "355f042b9228", "1002d44b513e", "155746dac788", "537774a8509d"]
#Mick, Mick, Craig, Carlo, Carlo, Carlo, Sara, Emanuele

def get_ldtrait_singleSNP(snp, pop="CEU+TSI+FIN+GBR+IBS", r2_d="r2", r2_d_threshold="0.1", window="500000", genome_build="grch37", tokens=TOKENS, sp2=None, use_logger=False, max_retries=1000):
    token = random.choice(tokens) if len(tokens) > 1 else tokens[0] 
    url = f"https://ldlink.nih.gov/LDlinkRest/ldtrait?token={token}"

    headers = {
        "Content-Type": "application/json",
    }

    for _ in range(max_retries):      
        data = {
            "snps": snp, 
            "pop": pop, 
            "r2_d": r2_d, 
            "r2_d_threshold": r2_d_threshold, 
            "window": window, 
            "genome_build": genome_build
        }  
        
        response = requests.post(url, headers=headers, data=json.dumps(data), verify=True)

        if response.status_code == 200:
            resp_txt = response.text
            if "Concurrent API requests restricted" in resp_txt:
                wait_time = random.randint(1, 5)
                print(f"Request for SNP {snp} failed due to concurrent API requests restriction. Retrying after {wait_time} seconds...")
                if use_logger:
                    logging.error(f"Request for SNP {snp} failed due to concurrent API requests restriction. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
            elif resp_txt.startswith('{\n  "error"'): 
                print(f"Request for SNP {snp} failed with error: {resp_txt}")
                if "No entries in the GWAS Catalog are identified using the LDtrait search criteria" in resp_txt and sp2:
                    old_snp = snp
                    snp = sp2.pop(0).split("_")[0]
                    while not snp.startswith("rs") and len(sp2):
                        snp = sp2.pop(0).split("_")[0]
                    print(f"Trying with the first element of SP2: {snp}, instead of SNP: {old_snp}")
                    if use_logger:
                        logging.warning(f"Trying with the first element of SP2: {snp}, instead of SNP: {old_snp}")                    
                else:
                    return None
            else:
                data = StringIO(resp_txt)
                return pd.read_csv(data, sep='\t')
        elif response.status_code in {408, 503, 504}:
            wait_time = random.randint(1, 5)
            print(f"Request for SNP {snp} timed-out with code {response.status_code}. Retrying after {wait_time} seconds...")
            if use_logger:
                logging.error(f"Request for SNP {snp} timed-out with code {response.status_code}. Retrying after {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print(f"Request for SNP {snp} failed with status code {response.status_code}")
            if use_logger:
                logging.error(f"Request for SNP {snp} failed with status code {response.status_code}")
            return None
    
def get_ldtrait_multiSNP(snps, pop="CEU+TSI+FIN+GBR+IBS", r2_d="r2", r2_d_threshold="0.1", window="500000", genome_build="grch37", tokens=TOKENS, use_logger=False, max_retries=1000):
    if len(snps) > 50:
        print(f"LDLink_LDTrait request can be for a maximum of 50 SNPs. Requested SNPs: {len(snps)}. Breaking the request into multiple requests...")
        if use_logger:
            logging.warning(f"LDLink_LDTrait request can be for a maximum of 50 SNPs. Requested SNPs: {len(snps)}. Breaking the request into multiple requests...")
        dfs = []
        for i in range(0, len(snps), 50):
            df = get_ldtrait_multiSNP(snps=snps[i:i+50], pop=pop, r2_d=r2_d, r2_d_threshold=r2_d_threshold, window=window, genome_build=genome_build, tokens=tokens, use_logger=use_logger, max_retries=max_retries)
            if df is not None:
                dfs.append(df)
        return pd.concat(dfs) if dfs else None

    token = random.choice(tokens) if len(tokens) > 1 else tokens[0]    
    url = f"https://ldlink.nih.gov/LDlinkRest/ldtrait?token={token}"

    headers = {
        "Content-Type": "application/json",
    }

    data = {
        "snps": "\n".join(snps), 
        "pop": pop, 
        "r2_d": r2_d, 
        "r2_d_threshold": r2_d_threshold, 
        "window": window, 
        "genome_build": genome_build
    }

    for _ in range(max_retries):
        response = requests.post(url, headers=headers, data=json.dumps(data), verify=True)
        if response.status_code == 200:
            resp_txt = response.text
            if "Concurrent API requests restricted" in resp_txt:
                wait_time = random.randint(1, 5)
                print(f"Request for SNP {snps[0]} failed due to concurrent API requests restriction. Retrying after {wait_time} seconds...")
                if use_logger:
                    logging.error(f"Request for SNP {snps[0]} failed due to concurrent API requests restriction. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
            elif resp_txt.startswith('{\n  "error"'): 
                print(f"Request for SNP {snps[0]} failed with error: {resp_txt}")
                if "Maximum variant list is 50  RS numbers or coordinates" in resp_txt:
                    print(f"Maximum variant list is 50 happened For the loci with lead variant {snps[0]}, number of SNPs {len(snps)}")
                return None
            else:
                data = StringIO(resp_txt)
                df = pd.read_csv(data, sep='\t')
                return df
        elif response.status_code in {408, 503, 504}:
            wait_time = random.randint(1, 5)
            print(f"Request for SNP {snps[0]} timed-out with code {response.status_code}. Retrying after {wait_time} seconds...")
            if use_logger:
                logging.error(f"Request for SNP {snps[0]} timed-out with code {response.status_code}. Retrying after {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print(f"Request for SNP {snps[0]} failed with status code {response.status_code}")
            if use_logger:
                logging.error(f"Request for SNP {snps[0]} failed with status code {response.status_code}")
            return None

    # If it reaches here, it means all attempts have failed.
    print(f"All attempts to retrieve SNP {snps[0]} have failed after {max_retries} retries.")
    if use_logger:
        logging.error(f"All attempts to retrieve SNP {snps[0]} have failed after {max_retries} retries.")

    return None