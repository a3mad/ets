# recommendation.py
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# -------------------------
# Global variables
# -------------------------
nbrs = None
interaction_matrix_sparse = None
user_mapping = {}
reverse_user_mapping = {}
item_mapping = {}
reverse_item_mapping = {}

def load_recommendation_data(filepath: str):
    """
    Reads the CSV at `filepath` and trains a Nearest Neighbors model
    consistent with the logic in your pipeline.
    """
    global nbrs, interaction_matrix_sparse
    global user_mapping, reverse_user_mapping
    global item_mapping, reverse_item_mapping

    # 1. Read CSV into DataFrame
    df = pd.read_csv(filepath)

    # 2. (Optional) Map event -> weight
    #    If you want a binary matrix, replace df['event_weight'] below with np.ones(len(df))
    event_weights = {'view': 1, 'addtocart': 4, 'transaction': 15}
    df['event_weight'] = df['event'].map(event_weights).fillna(1)  # fallback = 1 if event missing

    # 3. Create user/item mappings
    unique_users = df['visitorid'].unique()
    unique_items = df['itemid'].unique()

    user_mapping          = {user: i for i, user in enumerate(unique_users)}
    reverse_user_mapping  = {i: user for user, i in user_mapping.items()}
    item_mapping          = {item: i for i, item in enumerate(unique_items)}
    reverse_item_mapping  = {i: item for item, i in item_mapping.items()}

    # 4. Add mapped indices to DataFrame
    df['user_index'] = df['visitorid'].map(user_mapping)
    df['item_index'] = df['itemid'].map(item_mapping)

    # 5. Build the (weighted) interaction matrix
    num_users = len(unique_users)
    num_items = len(unique_items)
    interaction_matrix_sparse = csr_matrix(
        (
            df['event_weight'],
            (df['user_index'], df['item_index'])
        ),
        shape=(num_users, num_items)
    )

    # 6. Train the Nearest Neighbors model
    interaction_matrix_normalized = normalize(interaction_matrix_sparse, norm='l2')
    nbrs = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
    nbrs.fit(interaction_matrix_normalized)

    print("Nearest Neighbors model trained successfully.")

def recommend_items(visitor_id: int, top_n=10):
    """
    Returns a list of recommended items (based on the trained Nearest Neighbors model)
    for the given `visitor_id`.
    """
    # If model not loaded/trained, return empty
    if nbrs is None:
        print("Error: Model is not loaded or trained.")
        return []

    # If the visitor is unknown, return empty
    if visitor_id not in user_mapping:
        print("Error: user is not found.")
        return []

    # 1. Convert real visitor_id -> user_idx
    user_idx = user_mapping[visitor_id]

    # 2. Get the k-nearest neighbor users
    distances, indices = nbrs.kneighbors(interaction_matrix_sparse[user_idx], n_neighbors=5)
    similar_users = indices.flatten()[1:]  # exclude the user itself

    # 3. Aggregate item interactions from these similar users
    recommended_item_scores = interaction_matrix_sparse[similar_users].sum(axis=0).A1
    top_item_indices = recommended_item_scores.argsort()[-top_n:][::-1]

    # 4. Convert back to item IDs
    recommended_items = [reverse_item_mapping[i] for i in top_item_indices]

    return recommended_items
