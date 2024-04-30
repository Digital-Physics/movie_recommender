import time
# from functools import wraps
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import faiss

# Here's a brief overview of how Facebook AI Similarity Search (FAISS) works and why it can be faster than traditional k-nearest neighbors (KNN) algorithms:
# Indexing: Faiss uses advanced indexing techniques to organize the vectors in a way that makes similarity search faster. One of the key techniques is the use of hierarchical data structures like the inverted file or the IVFADC (Inverted File with Approximate Distance Calculation). These structures allow Faiss to quickly narrow down the search space to a subset of vectors that are likely to be similar to the query vector.
# Quantization: Faiss can quantize (or compress) the vectors into a lower-dimensional space, which reduces the memory footprint and speeds up the distance computations. This is particularly useful for high-dimensional vectors where the distance calculations can be computationally expensive.
# GPU Acceleration: Faiss provides GPU-accelerated implementations of the indexing and search algorithms, which can significantly speed up the search process, especially for large datasets and high-dimensional vectors.
# Efficient Distance Computations: Faiss uses optimized algorithms for computing distances between vectors, such as the inner product (for cosine similarity) or L2 distance. These algorithms are carefully optimized to take advantage of modern CPU and GPU architectures.
# we are using faiss-cpu, so will we see a speed-up?
# we are only using ~10k vectors. Is that big enough to see speed-up?
# we are only using an embedding dimension of 20. Is that big enough to see speed-up?

# Load the MovieLens dataset
ratings = pd.read_csv('../data/ratings.csv')
movies = pd.read_csv('../data/movies.csv')
# to do: make more fake data to compare KNN and FAISS when the number of movies/items is 100_000_000_000

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
Q = svd.fit_transform(X.T)

print("Shape of Q, our movie embeddings:", Q.shape)

# Index the item embeddings using Faiss
index = faiss.IndexFlatL2(20)  # L2 (Euclidean) distance with 20 dimensions
index.add(Q)

# Example query. Toy Story is movieId == 1 which gets mapped to index 0
query_embedding = svd.transform(X.T[movie_mapper[1]].reshape(1, -1))[0]
# query_embedding = svd.transform(X.T[movie_mapper[39]].reshape(1, -1))[0]
# D = distance, I = index
D, I = index.search(np.array([query_embedding]), k=10)
# even though the embeddings are in a 20-dimensional space, the result of the nearest neighbor search (D and I) does not directly reflect this; 
# The vector index returned does not store the original embeddings explicitly. 
print(f"{I.shape=}{I[0].shape=}{D[0][0].shape=}")

# Print top 10 similar movies to "Toy Story (1995)"
print("Top 10 similar movies to Toy Story (1995):")
for movie_idx in I[0]:
    print(movies[movies['movieId'] == movie_inv_mapper[movie_idx]]['title'])
    # print(type(movies[movies['movieId'] == movie_inv_mapper[movie_idx]]['title']))
    # print(movies[movies['movieId'] == movie_inv_mapper[movie_idx]]['title'].shape)
    # print(type(movies[movies['movieId'] == movie_inv_mapper[movie_idx]]['title'].iloc[0]))
    # iloc[0] is returning the single string object in the pandas data series of size (1,)
    title = movies[movies['movieId'] == movie_inv_mapper[movie_idx]]['title'].iloc[0]
    print(title)


print(f"{query_embedding.shape=}")
query_point = query_embedding.reshape(1, -1)
print(f"{query_point.shape=}")

# Timer decorator
def timer(func):
    # @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper

# Run the functions
@timer
def knn():
    knn = NearestNeighbors(n_neighbors=10, algorithm="brute", metric="euclidean")
    # knn.fit(X)
    knn.fit(Q)
    return knn.kneighbors(query_point, return_distance=False)

@timer
def approx_knn_faiss():
    # d = X.shape[1]
    d = Q.shape[1]
    index = faiss.IndexFlatL2(d)
    # index.add(X)
    index.add(Q)
    D, I = index.search(query_point, k=10)
    return I

@timer
def exact_knn():
    nn = NearestNeighbors(n_neighbors=10)
    # nn.fit(X)
    nn.fit(Q)
    return nn.kneighbors(query_point, return_distance=False)

print("K Nearest Neighbors:", knn())
print("Approximate K Nearest Neighbors with FAISS:", approx_knn_faiss())
print("Nearest Neighbors:", exact_knn())
