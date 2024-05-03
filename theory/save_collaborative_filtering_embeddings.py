import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import faiss

# Load the MovieLens dataset
ratings = pd.read_csv('../data/ratings.csv')
movies = pd.read_csv('../data/movies.csv')

# Create a sparse utility matrix
user_mapper = {val: i for i, val in enumerate(np.unique(ratings["userId"]))}
movie_mapper = {val: i for i, val in enumerate(np.unique(ratings["movieId"]))}
movie_inv_mapper = {i: val for i, val in enumerate(np.unique(ratings["movieId"]))}
user_index = [user_mapper[i] for i in ratings['userId']]
item_index = [movie_mapper[i] for i in ratings['movieId']]
print("utility matrix dimensions", len(user_mapper), len(movie_mapper))
X = csr_matrix((ratings["rating"], (user_index, item_index)), shape=(len(user_mapper), len(movie_mapper)))

# Factorize the Compressed Sparse Row represented utility matrix using TruncatedSVD
svd = TruncatedSVD(n_components=20, n_iter=10)
u = svd.fit_transform(X) # users embeddings, which is the left singular vectors matrix
v = svd.components_.T  # item embeddings, which is the right singular vectors matrix
# we aren't bringing over the singular values which relate to scale

print("Shape of our user embeddings:", u.shape)
print("Shape of our movie embeddings:", v.shape)

# Index the user embeddings using Faiss
index = faiss.IndexFlatL2(20)  # L2 (Euclidean) distance with 20 dimensions
index.add(u)

# Index the item embeddings using Faiss
index2 = faiss.IndexFlatL2(20)  # L2 (Euclidean) distance with 20 dimensions
index2.add(v)

# Save the embeddings to be imported into another python file
# Save the Faiss index to a file
faiss.write_index(index, "../embeddings/user_embeddings.index")
faiss.write_index(index2, "../embeddings/item_embeddings.index")

