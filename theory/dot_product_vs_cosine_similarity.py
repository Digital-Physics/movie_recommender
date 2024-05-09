import numpy as np
from numpy.linalg import norm  # Euclidean distance

# Define two sample vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([4*2, 5*2, 6*2])
d = np.array([-4*2, -5*2, -6*2])

dot_product = np.dot(a, b)
print("Dot Product:", dot_product)
dot_product2 = np.dot(a, c)
print("Dot Product:", dot_product2)
dot_product3 = np.dot(a, d)
print("Dot Product:", dot_product3)

print("The Dot Product is influence by the angle between the vectors AND THE MAGNITUDE of the vectors")
print()

# Cosine Similarity 
cosine_similarity = np.dot(a, b) / (norm(a) * norm(b))
print("Cosine Similarity:", cosine_similarity)
cosine_similarity2 = np.dot(a, c) / (norm(a) * norm(c))
print("Cosine Similarity:", cosine_similarity2)
cosine_similarity3 = np.dot(a, d) / (norm(a) * norm(d))
print("Cosine Similarity:", cosine_similarity3)

print("The Cosine Similarity is only influenced by the angle between the vectors")
print("Orthogonal = 0; 0 angle = 1; Opposite = -1")