import time
from Bio import Entrez
import xmltodict
from urllib.error import HTTPError
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import numpy as np

class dbSNP:
    def __init__(self, email="contact@soumick.com", api_key="0d61fd5ecedc0973699dff48a66936a36c09", retmax=20):
        Entrez.email = email
        Entrez.api_key = api_key
        self.retmax = retmax

    def get_rsIDs(self, data):
        results = []
        for row in data:
            chr, pos, allele0, allele1 = row
            eShandle = Entrez.esearch(db="snp",  
                                      term=f"{pos}[POSITION_GRCH37] AND {chr}[CHR]", 
                                      usehistory="y", 
                                      retmax=self.retmax 
                                     )
            eSresult = Entrez.read(eShandle)
            webenv = eSresult["WebEnv"]
            total_count = int(eSresult["Count"])
            query_key = eSresult["QueryKey"]

            for start in range(0, min(total_count, self.retmax)):
                fetch_handle = None
                for attempt in range(1, 4):
                    try:
                        fetch_handle = Entrez.efetch(db="snp",
                                                    retmode="xml",
                                                    retstart=start,
                                                    retmax=self.retmax,
                                                    webenv=webenv,
                                                    query_key=query_key )
                    except HTTPError as err:
                        if 500 <= err.code <= 599:
                            print(f"Received error from server {err}")
                            print("Attempt %i of 3" % attempt)
                            time.sleep(15)
                        else:
                            continue
                if (fetch_handle):
                    try:
                        data = xmltodict.parse(fetch_handle.read().decode("utf-8"))  
                        fetch_handle.close()                
                        if type(data['ExchangeSet']['DocumentSummary']) == list:
                            for i in range(len(data['ExchangeSet']['DocumentSummary'])):
                                _, _, al0, al1 = data['ExchangeSet']['DocumentSummary'][i]['SPDI'].split(":")
                                if (allele0 == al0 and allele1 == al1) or (allele0 == None and allele1 == None):
                                    return f"rs{data['ExchangeSet']['DocumentSummary'][i]['SNP_ID']}"
                        else:
                            _, _, al0, al1 = data['ExchangeSet']['DocumentSummary']['SPDI'].split(":")
                        if (allele0 == al0 and allele1 == al1) or (allele0 == None and allele1 == None):
                            return f"rs{data['ExchangeSet']['DocumentSummary']['SNP_ID']}"
                    except:
                        pass
        return None

db = dbSNP()
df_sumstats = pd.read_table("/project/ukbblatent/Out/Results/F20208_heart_1Ses_time2slc_MskCrop128_V2_BF16_3D_ph1lat2_4ChTrans128fold0_precbf16-mixed_pythaemodel-custom_ultra_vae/GWAS_fullDS/WBRIT_time2slc_Msk_V2_3D_L1_128FVAE_fullDS_ph1l/results/gwas/Z27.gwas.regenie.gz")
no_rs_id = df_sumstats[~df_sumstats["ID"].str.startswith("rs")]
data = no_rs_id[["CHROM", "GENPOS", "ALLELE0", "ALLELE1"]].values.tolist()

# Process data in chunks
chunk_size = 500  # Maximum number of IDs per request
chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

# Use ThreadPool for concurrent requests
pool = ThreadPool(processes=32)  # Number of threads
results = list(tqdm(pool.imap_unordered(db.get_rsIDs, chunks), total=len(chunks)))
pool.close()
pool.join()

# Flatten list of lists
results = [item for sublist in results for item in sublist]

mappings = {
    "original_key": no_rs_id["ID"].values.tolist(),
    "rsID": results
}

df_mappings = pd.DataFrame(mappings)
