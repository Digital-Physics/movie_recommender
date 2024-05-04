import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Note: Finding similar movie reviews != Finding similar movies to recommend
# Related Note: We don't have movie names for the reviews. Just text and labels (pos, neg).
# Bag of Words may work better for recommending similar articles

# Download the movie review corpus and stopwords
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt') # for tokenization

# Bag of Words is like an article/text embedding:
# each text has a sparse vocab frequency vector 
# we can ignore common words (stop words) and reduce the importance of terms that are common (TF-IDF)
# finally, the vector can be normalized to make sure the distance metrics aren't influenced by the text length

# Load the movie reviews and labels
reviews = []
labels = []

for fileid in movie_reviews.fileids():
    review = movie_reviews.raw(fileid)
    reviews.append(review)
    labels.append(fileid.split('/')[0])

# print(len(reviews))
# print(reviews[0])
# print(reviews[1])
# print(len(labels))
# print(labels[0])

# Tokenize the reviews and remove stop words 
# stop words include: a, an, and, are, as, at, be, but, by, etc.
stop_words = set(stopwords.words('english'))
# print(stop_words)

tokenized_reviews = []

for review in reviews:
    words = word_tokenize(review.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    tokenized_reviews.append(' '.join(words))

# Calculate term-frequency*inverse-document-frequency TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_reviews)

# Normalize the TF-IDF matrix
tfidf_normalized = normalize(tfidf_matrix)

# Find the index of Terminator reviews in the labels list
# specific_movie_indices = [i for i, label in enumerate(labels) if 'pos' in label and 'halloween' in review]
specific_movie_indices = [i for i, review in enumerate(reviews) if 'toy story' in review]
print(specific_movie_indices)

# Calculate cosine similarity between all 17 reviews that contained "toy story" and all other reviews
similarities = cosine_similarity(tfidf_normalized[specific_movie_indices], tfidf_normalized)
print(len(similarities), similarities[0])

# Find the most similar reviews
num_similar_reviews = 3 # per each toy story review (we have 17 query points)
unique_review_idx = set()

for i, similarities_to_review_i in enumerate(similarities):
    # Sort indices by similarity, excluding the original query point (a review tha) itself
    similar_indices = sorted(range(len(similarities_to_review_i)), key=lambda x: similarities_to_review_i[x], reverse=True)[1:num_similar_reviews+1]
    # print(f"Reviews similar to reviews w/ this 'toy story' review {i}:")
    for idx in similar_indices:
        unique_review_idx.add(idx)

reviews_to_try = unique_review_idx - set(specific_movie_indices)

for idx in reviews_to_try:
    print()
    print(idx)
    print(reviews[idx])

print(sorted(unique_review_idx))
print(sorted(specific_movie_indices))
print(sorted(reviews_to_try))


