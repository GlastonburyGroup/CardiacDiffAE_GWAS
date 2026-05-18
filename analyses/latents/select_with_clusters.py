# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score

def run_non_parametric_clustering(latent_sets):
    sorted_sizes = sorted(latent_sets.keys())
    
    UPPER_CAPACITY = sorted_sizes[-1]  # Use the largest latent size as upper capacity
    
    results_size = []
    results_score = []
    results_k = []
    
    print(f"\n{'='*20} DIRICHLET PROCESS (DP-GMM) ANALYSIS {'='*20}")
    print(f"{'Size':<6} | {'Natural k':<10} | {'Silhouette Score':<20}")
    print("-" * 50)
    
    for s in sorted_sizes:
        data = latent_sets[s]
        
        dpgmm = BayesianGaussianMixture(
            n_components=UPPER_CAPACITY,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=1.0/UPPER_CAPACITY, # Non-informative prior
            max_iter=500,
            n_init=3,
            random_state=42
        )
        
        dpgmm.fit(data)
        
        # Predict labels
        labels = dpgmm.predict(data)
        
        # --- Determine Effective k ---
        # The model "kills" unnecessary clusters by giving them weight ~ 0.
        # We count how many unique labels were actually assigned to data points.
        active_labels = np.unique(labels)
        n_active_clusters = len(active_labels)
        
        # If the model collapses everything to 1 cluster, Silhouette is undefined (-1)
        if n_active_clusters < 2:
            score = -1.0
        else:
            score = silhouette_score(data, labels)
            
        results_size.append(s)
        results_score.append(score)
        results_k.append(n_active_clusters)
        
        print(f"{s:<6} | {n_active_clusters:<10} | {score:<20.4f}")

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = '#16a085'
    ax1.set_xlabel('Latent Size')
    ax1.set_ylabel('Cluster Distinction (Silhouette)', color=color, fontsize=12)
    ax1.plot(results_size, results_score, 'o-', linewidth=2.5, color=color, label='Structure Quality')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Annotate with the discovered cluster counts
    for i, txt in enumerate(results_k):
        if results_score[i] > -0.5: # Only label valid scores
            ax1.annotate(f"k={txt}", (results_size[i], results_score[i]), 
                         xytext=(0, 10), textcoords='offset points', 
                         ha='center', fontsize=9, fontweight='bold')

    plt.title("Latent Space Selection via Non-Parametric Clustering (DP-GMM)")
    plt.tight_layout()
    plt.show()

    # --- Final Decision ---
    # We pick the size with the highest Silhouette score.
    # This size represents the clearest biological separation.
    
    best_idx = np.argmax(results_score)
    best_size = results_size[best_idx]
    
    print(f"\n{'-'*60}")
    print(f"FINAL DECISION: Size {best_size}")
    print(f"Reason: The data naturally separates into {results_k[best_idx]} distinct groups")
    print(f"with the highest structural clarity (Score: {results_score[best_idx]:.3f}).")
    print(f"{'-'*60}")
    
    return best_size

latent_sets = {}
latent_sets[128] = pd.read_table('/group/glastonbury/emma/data_segmentation_liver_shmolli/1_seed/QN_Latents/qn_latents_liver_FITPARAM_seg_128_1_seed_filt_ancestry.tsv', index_col=[0,1]).to_numpy()
latent_sets[64] = pd.read_table('/group/glastonbury/emma/data_segmentation_liver_shmolli/1_seed/QN_Latents/qn_latents_liver_FITPARAM_seg_64_1_seed_filt_ancestry.tsv', index_col=[0,1]).to_numpy()
latent_sets[32] = pd.read_table('/group/glastonbury/emma/data_segmentation_liver_shmolli/1_seed/QN_Latents/qn_latents_liver_FITPARAM_seg_32_1_seed_filt_ancestry.tsv', index_col=[0,1]).to_numpy()
latent_sets[16] = pd.read_table('/group/glastonbury/emma/data_segmentation_liver_shmolli/1_seed/QN_Latents/qn_latents_liver_FITPARAM_seg_16_1_seed_filt_ancestry.tsv', index_col=[0,1]).to_numpy()
latent_sets[8] = pd.read_table('/group/glastonbury/emma/data_segmentation_liver_shmolli/1_seed/QN_Latents/qn_latents_liver_FITPARAM_seg_8_1_seed_filt_ancestry.tsv', index_col=[0,1]).to_numpy()

# Run Analysis
best_size = run_non_parametric_clustering(latent_sets)
print(f"Selected best latent size: {best_size}")
