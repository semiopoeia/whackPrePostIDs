# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 12:07:55 2025
figuring out fuzzy match on some whack ids
individuals were given open field to write ID# and Email
across two survey cycles, creating a data merge problem
but because the same info is more or less placed
we use embeddings of strings, compute similarity indices matching cases,
then greedy clustering specifying at most clusters with n=2
issue is that because it is greedy clustering and individuals may not complete both survey cycles
clearly different cases get placed in cluster, so the matching is good at the beginning
but degenerates as you move down with single cycle completers getting paired
@author: PWS5
"""
!pip install -U sentence-transformers git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
!pip install pandas openpyxl scikit-learn

from huggingface_hub import login
login("hf_<<enteryourowntoken>>")
#pull token at hugging face so i can use embedding from gemma

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
#bring in data isolate strings
whackIDs=pd.read_excel("C:/IDs_whack.xlsx",header=None)
print(whackIDs.columns.tolist())
print(whackIDs.head())
strings = whackIDs[0].astype(str).tolist()
print(f"Loaded {len(strings)} strings.")

#embedding
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/embeddinggemma-300M"
model = SentenceTransformer(model_id).to(device=device)
embeddings = model.encode(strings, show_progress_bar=True)
print(f"Generated embeddings with shape: {embeddings.shape}")
#put in original data
whackIDs['Embedding'] = list(embeddings)

#calculate cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)
#set diagonal to a very low number so a string doesn't pair with itself
np.fill_diagonal(similarity_matrix, -1.0)

num_strings = len(strings)
paired_indices = set()
clusters = []
cluster_id = 0

print("Starting greedy pairing...")
while len(paired_indices) < num_strings:
    
#identify available individuals not yet paired
available_indices = [i for i in range(num_strings) if i not in paired_indices]
      if not available_indices:
        break # Should not happen if the loop condition is correct
 #sub-matrix for holding available cases for matching
    available_mask = np.isin(np.arange(num_strings), available_indices)
    sub_sim_matrix = similarity_matrix[available_mask][:, available_mask]
    
#allow singleton cluster when pairing is exhausted
    if len(available_indices) == 1:
        i = available_indices[0]
        clusters.append({
            'Cluster_ID': cluster_id, 
            'Strings': [strings[i]], 
            'Indices': [i],
            'Size': 1
        })
        paired_indices.add(i)
        cluster_id += 1
        break # All done
        
#match case to maximum similarity in the available pair sub-matrix
    max_sim_flat_index = np.argmax(sub_sim_matrix)
    
#convert flat index (1D embedding vector) back to 2D indices with rows and columns (relative to sub-matrix)
#max_sim_flat_index / len(available_indices) is integer division (row index)
#max_sim_flat_index % len(available_indices) is modulo (column index)
    idx_1_rel = max_sim_flat_index // len(available_indices)
    idx_2_rel = max_sim_flat_index % len(available_indices)
    
#convert relative indices back to original string indices
    idx_1 = available_indices[idx_1_rel]
    idx_2 = available_indices[idx_2_rel]
    
#similarity value
    similarity_value = similarity_matrix[idx_1, idx_2]
    
#form a new cluster
    clusters.append({
        'Cluster_ID': cluster_id,
        'Strings': [strings[idx_1], strings[idx_2]],
        'Indices': [idx_1, idx_2],
        'Similarity': similarity_value,
        'Size': 2
    })
    
#indicate cases as paired
    paired_indices.add(idx_1)
    paired_indices.add(idx_2)
    
#greedy match to prevent cases from being picked again by setting their similarity to -1
    similarity_matrix[idx_1, :] = -1.0
    similarity_matrix[:, idx_1] = -1.0
    similarity_matrix[idx_2, :] = -1.0
    similarity_matrix[:, idx_2] = -1.0
    
    cluster_id += 1

print("\n--- Clustering Results ---")
for cluster in clusters:
    print(f"Cluster {cluster['Cluster_ID']} (Size: {cluster['Size']}):")
    for s in cluster['Strings']:
        print(f"  - {s}")
    if cluster['Size'] == 2:
        print(f"  (Similarity: {cluster['Similarity']:.4f})")
    print("-" * 20)

#add cluster ID back to the main DataFrame
cluster_map = {}
for cluster in clusters:
    for idx in cluster['Indices']:
        cluster_map[idx] = cluster['Cluster_ID']

whackIDs['Cluster_ID'] = whackIDs.index.map(cluster_map)

#exporting data
output_file_name = 'C:/whackIDsClustered.xlsx'
# index=False prevents pandas from writing the row indices as an extra column
whackIDs.to_excel(output_file_name, index=False) 
