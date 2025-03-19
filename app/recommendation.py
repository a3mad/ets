import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# Global storage for recommendation data
recommendation_data = None
user_mapping = {}
reverse_user_mapping = {}
item_mapping = {}
reverse_item_mapping = {}
interaction_matrix_sparse = None
nbrs = None

def load_recommendation_data(df):
    """Processes the uploaded CSV to build the interaction matrix and train the recommendation model."""
    global recommendation_data, user_mapping, reverse_user_mapping, item_mapping, reverse_item_mapping, interaction_matrix_sparse, nbrs

    recommendation_data = df.copy()

    # Define event weights
    event_weights = {'view': 1, 'addtocart': 4, 'transaction': 15}
    df['event_weight'] = df['event'].map(event_weights)

    # Map visitorid and itemid to numerical indices
    unique_users = df['visitorid'].unique()
    unique_items = df['itemid'].unique()

    user_mapping = {user: i for i, user in enumerate(unique_users)}
    reverse_user_mapping = {i: user for user, i in user_mapping.items()}
    item_mapping = {item: i for i, item in enumerate(unique_items)}
    reverse_item_mapping = {i: item for item, i in item_mapping.items()}

    # Apply mappings
    df['user_index'] = df['visitorid'].map(user_mapping)
    df['item_index'] = df['itemid'].map(item_mapping)

    # Build sparse interaction matrix (weighted)
    num_users, num_items = len(unique_users), len(unique_items)
    interaction_matrix_sparse = csr_matrix(
        (df['event_weight'], (df['user_index'], df['item_index'])),
        shape=(num_users, num_items)
    )

    # Train Nearest Neighbors model
    interaction_matrix_normalized = normalize(interaction_matrix_sparse, norm='l2')
    nbrs = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
    nbrs.fit(interaction_matrix_normalized)

    print("Recommendation model trained successfully.")

def recommend_items(visitorid):
    """Generates item recommendations for a given visitor using Nearest Neighbors."""
    if visitorid not in user_mapping:
        return []

    user_idx = user_mapping[visitorid]
    distances, indices = nbrs.kneighbors(interaction_matrix_sparse[user_idx], n_neighbors=5)

    similar_users = indices.flatten()[1:]  # Exclude the user itself
    recommended_items = interaction_matrix_sparse[similar_users].sum(axis=0).A1.argsort()[-10:][::-1]

    return [reverse_item_mapping[i] for i in recommended_items if i in reverse_item_mapping]