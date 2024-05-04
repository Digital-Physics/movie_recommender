from torch import nn
import torch
import pandas as pd
import faiss


###### Serving in Production
# should we even import a model with the item embedding tower? if they change, yes. if not, no.
# the user and item embeddings exist in the same space 🧐
# we can do retrieval on whatever the user embedding is coming out of it's tower
# https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture
# Step 4: Evaluate the performance with A/B tests. Use the new model on a subset of users before evaluating whether it should replace our current default model in Production.

# Step 4a: FAISS retrieval to find the approximate 250 closest vector candidates (we only have ~10k movies, but let's pretend there were billions). These might be stored in a vector DB.
index = faiss.read_index("./embeddings/two_tower_item_embeddings.index")
TwoTowerModel = torch.load("./trained_model/two_tower_model.pth")

# dimensions for initializing TwoTowerModel:
# we have two separate Bag of Words for user search history and item text. both are the same length; should we use a more sophisticated embedding for text?
bag_of_words_embedding_length = 32 # reduced from vocab length to 32 using PCA or t-SNE or some other method
last_item_click = 64 # same as the embedding size at the top of our towers
item_feature_size = len(["post_age", "likes_per_hour", "user_id"]) + bag_of_words_embedding_length
user_feature_size = len(["user_age", "geo_location", "logins_per_week"]) 
context_feature_size = last_item_click + bag_of_words_embedding_length

two_tower_model_instance = TwoTowerModel(item_feature_size, user_feature_size, context_feature_size)

right_tower_item_features = torch.randn(250, item_feature_size)
left_tower_user_features = torch.randn(user_feature_size)
context_features = torch.randn(context_feature_size)

# 250 item candidates for one user
input_tuples_for_candidate_items = zip(right_tower_item_features, left_tower_user_features * 250, context_features * 250)

# Step 4b: Rank the 250 candidate items for user j by doing 250 forward passes on our trained two tower model (inference) 
predictions = (two_tower_model_instance(a, b, c) for a, b, c in input_tuples_for_candidate_items)

inputs_and_predictions = zip(input_tuples_for_candidate_items, predictions)
inputs_and_predictions.sort(reverse=True, key=lambda x: x[-1])
print(inputs_and_predictions)

# Step 4c: Post-process the rankings. Select for diversity of recommendations, filter content you don't want, etc. which is essentially re-ranking
# to do: put some logic in here once we get some better fake data

# Step 5-6: Update the AUC and other evaluation metrics with new data
# to do: simulate choices on the recommended items for many users (we are just going through the recommendation ranking for one user an one point in time at the moment)
# We could include Area under the ROCurve. Greater 70% or greater than current model could be threshold. Make sure threshold > majority class percentage
# If we are using a regression metric and not a binary classification, we could look at MSE, R**2

# Step 5-6: Log raw user interaction data. Collect data for the next two tower re-training. Raw data will allow are features to evolve if we want to change something in the future.
# Note: we should log raw user data (w/ timestamps) so our features can evolve over versions of the recommender model, if needed.
# We start off using historical data to train our model, but we'd like to fold in our new data and retrain our model as we get new user engagement with our recommendations
# we don't want to lose information by assuming all we need to capture is the features we are currently using.

# Step 7: Update (context) features for the user which change based on last_item_clicks. Update item features if a video got a like or something.
# This logic actually feels like it should be part of the UI design

# Step 8: Periodically or continuously retrain two tower model

# Thoughts to consider: 
# Serving in production.
# Latency?: How quickly can retrieval and inference be done? Can we score all the candidate items in parallel?
# Scalability?: Can we scale servers (horizontally) as we get more items and users? Do our algorithms scale? 
# Retraining?: What is our online learning approach or pipelines for retraining?

