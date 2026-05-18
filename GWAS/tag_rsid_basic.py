import time
from Bio import Entrez
import xmltodict
from urllib.error import HTTPError
import pandas as pd
from tqdm import tqdm
class dbSNP:
    def __init__(self, email="contact@soumick.com", api_key="0d61fd5ecedc0973699dff48a66936a36c09", retmax=20):
        Entrez.email = email
        Entrez.api_key = api_key
        self.retmax = retmax

    def get_rsID(self, chr, pos, allele0=None, allele1=None):
        # print(f"Searching for {chr}:{pos} {allele0}/{allele1}")
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
            end = min(total_count, start+self.retmax)
            fetch_handle = None
            for attempt in range(1, 4):
                try:
                    fetch_handle = Entrez.efetch(db="snp",
                                                #rettype="uilist", #available types [uilist | xml (use retmode=xml))
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
                
        return None

db = dbSNP()
db.get_rsID("1", "64402765")

df_sumstats = pd.read_table("/project/ukbblatent/Out/Results/F20208_heart_1Ses_time2slc_MskCrop128_V2_BF16_3D_ph1lat2_4ChTrans128fold0_precbf16-mixed_pythaemodel-custom_ultra_vae/GWAS_fullDS/WBRIT_time2slc_Msk_V2_3D_L1_128FVAE_fullDS_ph1l/results/gwas/Z27.gwas.regenie.gz")
no_rs_id = df_sumstats[~df_sumstats["ID"].str.startswith("rs")]

mappings = {
    "original_key": [],
    "rsID": []
}

for index, row in tqdm(no_rs_id.iterrows(), total=no_rs_id.shape[0]):
    rsID = db.get_rsID(row["CHROM"], row["GENPOS"], row["ALLELE0"], row["ALLELE1"])
    if rsID is None:
        continue
    mappings["original_key"].append(row["ID"])
    mappings["rsID"].append(rsID)

df_mappings = pd.DataFrame(mappings)
df_mappings.to_csv("/group/glastonbury/soumick/scratch/GWAS/rsID_dbSNP_mappings.tsv", index=False, sep="\t")