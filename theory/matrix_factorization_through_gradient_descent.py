import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
import faiss
import pandas as pd
import numpy as np

# Load the data
ratings = pd.read_csv('../data/ratings.csv')
movies = pd.read_csv('../data/movies.csv')

movie_titles = dict(zip(movies['movieId'], movies['title']))

# Create a sparse utility matrix
user_mapper = {val: i for i, val in enumerate(np.unique(ratings["userId"]))}
movie_mapper = {val: i for i, val in enumerate(np.unique(ratings["movieId"]))}
movie_inv_mapper = {i: val for i, val in enumerate(np.unique(ratings["movieId"]))}
user_index = [user_mapper[i] for i in ratings['userId']]
item_index = [movie_mapper[i] for i in ratings['movieId']]
print("utility matrix dimensions", len(user_mapper), len(movie_mapper))
X = csr_matrix((ratings["rating"], (user_index, item_index)), shape=(len(user_mapper), len(movie_mapper)))

# Convert to PyTorch tensors
X = torch.tensor(X.toarray(), dtype=torch.float32)

# Split the data into training and cross-validation sets
X_train, X_cv = train_test_split(X, test_size=0.2, random_state=42)


# Matrix factorization using gradient descent
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        return torch.sum(user_embedding * item_embedding, dim=1)

# Hyperparameters
embedding_size = 20
num_epochs = 1000
learning_rate = 0.2  
k = 10  # Define the number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Model and optimizer
model = MatrixFactorization(X.shape[0], X.shape[1], embedding_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Iterate over each fold
for fold, (train_index, cv_index) in enumerate(kf.split(X)):
    X_train, X_cv = X[train_index], X[cv_index]

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        non_zero_indices = torch.nonzero(X_train)
        user_indices = non_zero_indices[:, 0]
        item_indices = non_zero_indices[:, 1]
        user_indices = torch.tensor(user_indices, dtype=torch.long)
        item_indices = torch.tensor(item_indices, dtype=torch.long)
        ratings = X_train[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        predictions = model(user_indices, item_indices)
        loss = nn.MSELoss()(predictions, ratings)
        loss.backward()
        optimizer.step()

        # Compute loss on the cross-validation set
        with torch.no_grad():
            non_zero_indices_cv = torch.nonzero(X_cv)
            user_indices_cv = non_zero_indices_cv[:, 0]
            item_indices_cv = non_zero_indices_cv[:, 1]
            user_indices_cv = torch.tensor(user_indices_cv, dtype=torch.long)
            item_indices_cv = torch.tensor(item_indices_cv, dtype=torch.long)
            ratings_cv = X_cv[non_zero_indices_cv[:, 0], non_zero_indices_cv[:, 1]]
            predictions_cv = model(user_indices_cv, item_indices_cv)
            loss_cv = nn.MSELoss()(predictions_cv, ratings_cv)

        print(f"Fold {fold+1}/{k}, Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, CV Loss: {loss_cv.item()}")

# Extract the item embeddings
item_embeddings = model.item_embedding.weight.detach().numpy()

# Use Faiss to find the k-nearest neighbors for a particular movie
query_movie_id = 0  # Example movie ID (Toy Story)
query_embedding = item_embeddings[query_movie_id]
print(f"{query_embedding=}")
query_point = query_embedding.reshape(1, -1)
d = query_embedding.shape[0]
index = faiss.IndexFlatL2(d)
index.add(item_embeddings)
distance, index = index.search(query_point, k=10)

# Convert indices to movie IDs
nearest_neighbors_indices = [movie_inv_mapper[i] for i in index[0]]
nearest_neighbors_distances = distance[0]

# Print the nearest neighbors
for neighbor_idx, dist in zip(nearest_neighbors_indices, nearest_neighbors_distances):
    print(f"Movie ID: {neighbor_idx}, Movie: {movie_titles[neighbor_idx]}, Distance: {dist}")
