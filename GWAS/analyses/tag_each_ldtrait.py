import argparse
import pandas as pd

try:
    from utils.find_ols_path import get_ontology_forrest, find_term
    from utils.nlp import gen_wordcloud
except:
    from .utils.find_ols_path import get_ontology_forrest, find_term
    from .utils.nlp import gen_wordcloud

mapping = {
    "NA": "Other traits, not in our list of OLSs",
    "EFO:0003777": "EFO_0003777 : heart disease (part of 319)",
    "EFO:0004294": "EFO_0004294 : left atrial function",
    "EFO:0004295": "EFO_0004295 : left ventricular function",
    "EFO:0009463": "EFO_0009463 : infarction",
    "EFO:0000318": "EFO_0000318 : cardiomiopathy (part of 319)",
    "EFO:0000319": "EFO_0000319 : cardiovascular disease",
    "EFO:0009506": "EFO_0009506 : heart injury",
    "EFO:1001339": "EFO_1001339 : heart neoplasm (part of 319)",
    "EFO:0002461": "EFO_0002461 : skeletal system disease",
    "EFO:0003923": "EFO_0003923 : bone density (part of 4512)",
    "EFO:0003931": "EFO_0003931 : bone fracture (part of 4512)",
    "EFO:0004260": "EFO_0004260 : bone disease",
    "EFO:0004512": "EFO_0004512 : bone measurement",
    "EFO:0004298": "EFO_0004298 : cardiovascular measurement",
    "EFO:0004503": "EFO_0004503 : hematological measurement",
    "EFO:0005105": "EFO_0005105 : lipid or lipoprotein measurement",
    "EFO:0004303": "EFO_0004303 : vital signs",
    "EFO:0003892": "EFO_0003892 : pulmonary function measurement",
    "EFO:0004324": "EFO_0004324 : body weights and measures",
    "EFO:0004764": "EFO_0004764 : visceral adipose tissue volumes",
    "EFO:0004464": "EFO_0004464 : brain measurement",
    
    "EFO:0004340": "EFO:0004340: BMI",
    "MONDO:0005148": "MONDO:0005148: type 2 diabetes mellitus",
    "EFO:0000275": "EFO:0000275: atrial fibrillation (part of 3777, as well as 319)",
    "EFO:1001375": "EFO:1001375: Myocardial Ischemia (part of 3777, as well as 319)",
    "EFO:0000612": "EFO:0000612: myocardial infarction (part of 3777, as well as 319)",
    "HP:0001627": "HP:0001627: Abnormal heart morphology",
    "HP:000164": "HP:000164: Cardiomegaly (part of HP:0001627)",
    "EFO:0000537": "EFO:0000537: hypertension (part of 319)",
    "EFO:0001645": "EFO:0001645: coronary artery disease (part of 3777, as well as 319)",
    "EFO:0009531": "EFO:0009531: aortic valve disease (part of 3777, as well as 319)",
    "EFO:0009564": "EFO:0009564: pulmonary valve disease (part of 3777, as well as 319)",
}

def get_counts(df, path, tag, mode='a'):
    df_expanded = df.assign(tags=df['tags'].str.split(',')).explode('tags')
    unique_leadSNP_counts_per_tag = df_expanded.groupby('tags')['leadSNP'].nunique()
    df_unique_leadSNP_counts_per_tag = unique_leadSNP_counts_per_tag.reset_index()
    df_unique_leadSNP_counts_per_tag.columns = ['tags', 'unique_SNPs']
    df_unique_leadSNP_counts_per_tag.sort_values(by="unique_SNPs", inplace=True, ascending=False)
    df_unique_leadSNP_counts_per_tag.tags = df_unique_leadSNP_counts_per_tag.tags.map(mapping)

    with open(path, mode) as f:
        f.write(f"\n\n{tag}:..............\n")
        df_unique_leadSNP_counts_per_tag.to_csv(f, sep='\t', index=False, mode=mode)

def tag_each_ldtrait(path, sig_level, non_root_interesting=None):
    if not bool(non_root_interesting):
        non_root_interesting = "EFO:0003777,EFO_0000318,EFO:1001339,EFO:0003923,EFO:0003931,EFO:0000275,EFO:1001375,EFO:0000612,HP:000164,EFO:0000537,EFO:0001645,EFO:0009531,EFO:0009564"

    if isinstance(non_root_interesting, str):
        non_root_interesting = non_root_interesting.split(",")

    df = pd.read_table(path)

    ontology_forrest = get_ontology_forrest()

    traits = df["GWAS Trait"].unique()
    for trait in traits:
        trait = trait.lower()
        trees = find_term(ontology_forrest, trait)
        roots = list({t[0] for t in trees})

        for nri in non_root_interesting:
            if any(nri in sublist for sublist in trees):
                roots.append(nri)

        tags = ','.join(set(roots))

        df.loc[df["GWAS Trait"].str.lower() == trait, "tags"] = tags or "NA"

    df.to_csv(path.replace("all.", "all.tagged."), sep="\t", index=False)

    df = df[df["P-value"] < sig_level]
    df_sig_R5 = df[df["R2"] >= 0.5]
    df_sig_R6 = df[df["R2"] >= 0.6]
    df_sig_R8 = df[df["R2"] >= 0.8]
    
    if other_traits := df[df["tags"] == "NA"]["GWAS Trait"].unique().tolist():
        with open(path.replace("all.ldtrait.tsv", "all.ldtrait.other_traits.txt"), "w") as f:
            f.write("Other traits, not in our list of OLSs, but significant LDLink_LDTrait:..........\n")
            f.write("\n".join(other_traits))

            if other_traits_sig_R5 := df_sig_R5[df_sig_R5["tags"] == "NA"]["GWAS Trait"].unique().tolist():
                f.write("\n\n\n\nOther traits, as per LDLink_LDTrait, they are significant and the R2 is greater than or equals to 0.5:..........\n")
                f.write("\n".join(other_traits_sig_R5))

            if other_traits_sig_R6 := df_sig_R6[df_sig_R6["tags"] == "NA"]["GWAS Trait"].unique().tolist():
                f.write("\n\n\n\nOther traits, as per LDLink_LDTrait, they are significant and the R2 is greater than or equals to 0.6:..........\n")
                f.write("\n".join(other_traits_sig_R6))

            if other_traits_sig_R8 := df_sig_R8[df_sig_R8["tags"] == "NA"]["GWAS Trait"].unique().tolist():
                f.write("\n\n\n\nOther traits, as per LDLink_LDTrait, they are significant and the R2 is greater than or equals to 0.8:..........\n")
                f.write("\n".join(other_traits_sig_R8))

        gen_wordcloud(df[df["tags"] == "NA"], df_is_indtraits=True, background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', filename=
        path.replace("all.ldtrait.tsv", "all.ldtrait.other_traits.png"))

    get_counts(df=df, path=path.replace("all.ldtrait.tsv", "all.ldtrait.ols.tagcounts.txt"), tag="# SNPs / Toploci with significant LDLink_LDTrait", mode='w')
    get_counts(df=df_sig_R5, path=path.replace("all.ldtrait.tsv", "all.ldtrait.ols.tagcounts.txt"), tag="# SNPs / Toploci with significant LDLink_LDTrait and the R2 is greater than or equals to 0.5", mode='a')
    get_counts(df=df_sig_R6, path=path.replace("all.ldtrait.tsv", "all.ldtrait.ols.tagcounts.txt"), tag="# SNPs / Toploci with significant LDLink_LDTrait and the R2 is greater than or equals to 0.6", mode='a')
    get_counts(df=df_sig_R8, path=path.replace("all.ldtrait.tsv", "all.ldtrait.ols.tagcounts.txt"), tag="# SNPs / Toploci with significant LDLink_LDTrait and the R2 is greater than or equals to 0.8", mode='a')

    #to get the unique SNPs per tag
    # unique_leadSNP_per_tag = df_expanded.groupby('tags')['leadSNP'].unique()
    # df_unique_leadSNP_per_tag = unique_leadSNP_per_tag.explode().reset_index()

def getARGSParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default="/project/ukbblatent/Out/Results/F20208_heart_1Ses_time2slc_MskCrop128_3pheno_V2_3D100ep_ph2ldo50_4ChTrans128fold0_precbf16-mixed_pythaemodel-custom_ultra_vae/GWAS_fullDS/WBRIT_time2slc_Msk_V2_3D100ep_L1_128VAE_fullDS_ph2ldo50", help='Location where the GWAS outputs are storred [The main folder, containing a subfolder results, containing subfolder gwas]')
    parser.add_argument('--non_root_interesting', type=str, default="EFO:0003777,EFO_0000318,EFO:1001339,EFO:0003923,EFO:0003931,EFO:0000275,EFO:1001375,EFO:0000612,HP:000164,EFO:0000537,EFO:0001645,EFO:0009531,EFO:0009564", help='Path where LDLink_LDtraits folder will be created and the outputs will be storred [If blank, it will be storred inside the analyses folder present in the parent folder of path2gwasout]')
    
    parser.add_argument("--sig_level", type=float, default=5e-8, help="Significance level.")
    
    return parser

if __name__ == "__main__":
    parser = getARGSParser()
    args, _ = parser.parse_known_args() 

    if not args.path.endswith("tsv"):
        args.path = f"{args.path}/results/analyses/LDLink_LDtraits/all.ldtrait.tsv"
    
    tag_each_ldtrait(args.path, args.sig_level, args.non_root_interesting)
    
    print("Done")
