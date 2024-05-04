import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import faiss

###### Offline training
# Step 1a: parse, clean & transform logged data (or Load & transform in our case)
ratings = pd.read_csv('./data/ratings.csv')
movies = pd.read_csv('./data/movies.csv')

# list of (key=movieId, value=title) tuples for lookup later
movie_titles = dict(zip(movies['movieId'], movies['title']))

# Create a sparse utility/interaction matrix. 
# map user and movie Ids to consecutive index numbers staring with 0
user_mapper = {val: i for i, val in enumerate(np.unique(ratings["userId"]))}
movie_mapper = {val: i for i, val in enumerate(np.unique(ratings["movieId"]))}
movie_inv_mapper = {i: val for i, val in enumerate(np.unique(ratings["movieId"]))}

print(ratings.columns)
# create the lists that will correspond to the list of ratings (or in this case the "engagement") in our user-item engagement matrix
user_index = [user_mapper[i] for i in ratings['userId']] # rows, user i
item_index = [movie_mapper[j] for j in ratings['movieId']] # cols, user j
engagement = [1 if rating >= 3.5 else 0 for rating in ratings["rating"]] # to mix it up, let's pretend the ratings are really engagement/no-engagement binary data

# y values, y indices (i, j) , shape of matrix
# y = csr_matrix((engagement, (user_index, item_index)), shape=(len(user_mapper), len(movie_mapper)))
# y = torch.tensor(y.toarray(), dtype=torch.float32)
# print("engagement matrix dimensions:", len(user_mapper), len(movie_mapper), y.shape)
print("Length of list of training example entries (i, j) in the sparse matrix:", len(engagement))
y = torch.tensor(engagement)

# We should have a good mix of both positive and negative examples engagement examples. let's print the distribution
print(f"positive examples: {sum(engagement)}")
print(f"negative examples: {len(engagement) - sum(engagement)}")

# Create fake user and item features:
# note: if we do any normalization, (technically) we should make sure we aren't having any minor data leaks from the test set (e.g. divide by average which includes test data) 
# to do: think of more/better features. 
# to do: consider whether we need item context features for the item... so both the item and user are a rep of their entire history and their their relevance to now/recent vs the past
# we have two separate Bag of Words for user search history and item text. both are the same length; should we use a more sophisticated embedding for text?
bag_of_words_embedding_length = 32 # reduced from vocab length to 32 using PCA or t-SNE or some other method
last_item_click = 64 # same as the embedding size at the top of our towers
# notice how our item embeddings will change over time. do we want to update these embeddings in FAISS at deployment time?
item_feature_size = len(["post_age", "likes_per_hour", "user_id"]) + bag_of_words_embedding_length
user_feature_size = len(["user_age", "geo_location", "logins_per_week"]) 
context_feature_size = last_item_click + bag_of_words_embedding_length

# Fake training data. 
# technically, better fake data would use the same random item features for the same user i or movie j each time in training
# this became apparent when we were ready to save the embeddings for each item by running it through its trained tower side
X_and_y = {
    "item_features": torch.randn(len(y), item_feature_size),
    "user_features": torch.randn(len(y), user_feature_size),
    "context_features": torch.randn(len(y), context_feature_size),
    "engagement": y
}

# Step 1b: feature engineering (already done)

# Step 1c: train and validation split 
# Skip here; we'll use kf.split() to get the indices in our for loop and then get our training and cv there
# Note: we won't have a 3rd test_set because we'll do K-Folds cross validation and use the CV metrics after the K-th round for the final evaluation)
# X_and_y_train, X_and_y_cv = train_test_split(*X_and_y.values(), test_size=0.2, random_state=42)

# Step 2: Model Architecture
class TwoTowerModel(nn.Module):
    def __init__(self, item_feature_size, static_user_feature_size, context_feature_size):
        """"this two tower model imports embeddings, but we may not have item embeddings (or user embeddings) from a collaborative filtering kind of process if new items don't have an embedding yet.
        also, we are assuming the output is a binary classifier (click/no-click) but we could have a regression model with a score (click = 1, comment = 2, like = 3, score = 4)."""
        # super() is used to call the constructor of the parent class of TwoTowerModel, which is nn.Module. 
        # This is necessary because TwoTowerModel is inheriting from nn.Module, and you need to initialize the nn.Module part of TwoTowerModel before adding your custom functionality.
        # When you inherit from a parent class, Python creates a new class that includes all the attributes and methods of the parent class. 
        # However, to properly initialize the parent class's attributes and set up its internal state, you need to explicitly call the parent class's constructor
        super(TwoTowerModel, self).__init__()

        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.user_tower = nn.Sequential(
            nn.Linear(static_user_feature_size + context_feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
    
    def forward(self, item_features, user_features, context_features, save_embeddings_flag=False):
        item_embedding = self.item_tower(item_features)
        if save_embeddings_flag:
            return item_embedding.detach()
        user_embedding = self.user_tower(torch.cat((user_features, context_features), dim=1))
        dot_product = torch.sum(item_embedding * user_embedding, dim=1, keepdim=True)
        return torch.sigmoid(dot_product)

# Step 3a: train a two tower ReLu Neural Net with a dot product at the last layer (followed by a sigmoid; Ground truth is 0 or 1).
# Note: we may also want to explore boosting architectures such as XGBoost, CatBoost, or LGBM after our Neural Net approach. 

# Hyperparameters
num_epochs = 5
learning_rate = 0.01
k_folds = 10  

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize Model and select optimizer
model = TwoTowerModel(item_feature_size, user_feature_size, context_feature_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for fold, (train_index, cv_index) in enumerate(kf.split(X_and_y["engagement"])):
    X_and_y_train = {
        "item_features": X_and_y["item_features"][train_index],
        "user_features": X_and_y["user_features"][train_index],
        "context_features": X_and_y["context_features"][train_index],
        "engagement": X_and_y["engagement"][train_index]
    }

    X_and_y_cv = {
        "item_features": X_and_y["item_features"][cv_index],
        "user_features": X_and_y["user_features"][cv_index],
        "context_features": X_and_y["context_features"][cv_index],
        "engagement": X_and_y["engagement"][cv_index]
    }

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad() # we'll calc the gradient for each param/weight again
        print(X_and_y_train["engagement"].shape)
        print(X_and_y_train["item_features"].shape, X_and_y_train["user_features"].shape, X_and_y_train["context_features"].shape)
        predictions = model(X_and_y_train["item_features"], X_and_y_train["user_features"], X_and_y_train["context_features"])
        ground_truth = X_and_y_train["engagement"]
        loss = nn.BCELoss()(predictions.view(-1),  ground_truth.view(-1).float())
        loss.backward()
        optimizer.step()

        # Compute loss on the cross-validation set
        with torch.no_grad():
            predictions_cv = model(X_and_y_cv["item_features"], X_and_y_cv["user_features"], X_and_y_cv["context_features"])
            ground_truth = X_and_y_cv["engagement"]
            loss_cv = nn.BCELoss()(predictions_cv.view(-1), ground_truth.view(-1).float())

        print(f"Fold {fold+1}/{k_folds}, Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, CV Loss: {loss_cv.item()}")

# Step 3b: Evaluate the model
# Normally, we'd want to look at some metrics like AUC, but we're using fake data here
# Note the imbalance in the dataset for this kind of analysis; you can always predict the majority class and be right that percentage of the time.

# Step 3c: Save the trained model
torch.save(model.state_dict(), "./trained_model/two_tower_model.pth")

# Step 3d: Save item embeddings (and user embeddings if we want them too) for the pre-ranking item candidate retrieval step
# but notice how our item embeddings will change over time. do we want to update these embeddings during serving time?
item_embeddings_list = []
dummy_user_features = torch.randn(item_feature_size) # we only care about saving the item embeddings for FAISS
dummy_context_features = torch.randn(context_feature_size) # we only care about saving the item embeddings for FAISS

# for item_features in X_and_y["item_features"]: # this has duplicate items because this is a training list
for item_features in (torch.randn(item_feature_size) for _ in range(len(movie_mapper))): # fix this after we get some real data 
    item_embeddings_list.append(model(item_features, dummy_user_features, dummy_context_features, True))

# Index the item embeddings using Faiss
index = faiss.IndexFlatL2(64)  # L2 (Euclidean) distance with 64 dimensions
tensor_of_embeddings = torch.stack(item_embeddings_list)
print(f"{tensor_of_embeddings.shape=}")
# FAISS will index these embedding vectors which will be one of the ways it does approximate nearest neighbor quicker
index.add(tensor_of_embeddings) 

# Save the embeddings to be imported into another python file for retrieval at inference time
faiss.write_index(index, "./embeddings/two_tower_item_embeddings.index")