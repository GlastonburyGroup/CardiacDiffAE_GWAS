import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

def finalise_latentDFs(in_npy_path, out_tsv_path, quantile_transform=False):
    embdf = create_latentdf_singleEmbPerSub(in_npy_path)
    latents = process_embdf(embdf)
    latents.insert(0, 'FID', latents.index)
    latents.insert(1, 'IID', latents.index)

    if quantile_transform:
        latents.to_csv(out_tsv_path.replace(".tsv", "_raw.tsv"), sep='\t', index=False)
        for col in latents.columns:
            if col in ["FID", "IID"]:
                continue
            normaliser = QuantileTransformer(output_distribution='normal')
            latents[col] = normaliser.fit_transform(np.expand_dims(latents[col], axis=1)).squeeze()

    latents.to_csv(out_tsv_path, sep='\t', index=False)
    print("Final latent TSV creation Done!")

# Read (NPY file) and prepare the latent embeddings
def create_latentdf_singleEmbPerSub(embpath="", processed_emb=None):
    if processed_emb is None:
        processed_emb = np.load(embpath, allow_pickle=True)
    
    processed_emb = pd.DataFrame(processed_emb, columns=['id', 'dataset', 'MRIvisit', 'data_tag', 'Zs'])
    processed_emb = processed_emb.set_index('id')
    processed_emb.index = processed_emb.index.astype(int)

    processed_emb.Zs = processed_emb.Zs.apply(np.squeeze)
    df = processed_emb.Zs.apply(pd.Series)
    df.columns = [f'Z{i}' for i in range(len(df.columns))]
    processed_emb = pd.concat([processed_emb.drop('Zs', axis=1), df], axis=1)
    processed_emb.MRIvisit = processed_emb.MRIvisit.apply(lambda x: str(x)[0]).astype('int')

    return processed_emb

def process_embdf(embdf):
    embdf2 = embdf[embdf['MRIvisit'] == 2]
    embdf3 = embdf[embdf['MRIvisit'] == 3]
    latents = embdf2.drop(['dataset', 'data_tag', 'MRIvisit'], axis=1)
    embdf_only3 = embdf3[~embdf3.index.isin(latents.index)]
    latents3 = embdf_only3.drop(['dataset', 'data_tag', 'MRIvisit'], axis=1)
    return pd.concat([latents, latents3])

### FOR simple FactorVAE models - not time2ch (1 latent vector per time point)
def create_latentdf_onetp(embpath, tp): #TODO: use it, not yet done
    processed_emb = np.load(embpath, allow_pickle=True)
    processed_emb = pd.DataFrame(processed_emb, columns=['id', 'dataset', 'MRIvisit', 'data_tag', 'Zs'])
    processed_emb = processed_emb.set_index('id')
    processed_emb.index = processed_emb.index.astype(int)

    processed_emb.Zs = processed_emb.Zs.apply(np.squeeze) 
    # take only one time point
    processed_emb['Zs_tp'] = processed_emb.Zs.apply(lambda x: x[tp, :]) 
    df = processed_emb.Zs_tp.apply(pd.Series)
    df.columns = [f'Z{i}' for i in range(len(df.columns))]
    df = pd.concat([processed_emb.drop(['dataset', 'data_tag', 'Zs', 'Zs_tp'], axis=1), df], axis=1)
    df.MRIvisit = df.MRIvisit.apply(lambda x: str(x)[0]).astype('int')
    return df