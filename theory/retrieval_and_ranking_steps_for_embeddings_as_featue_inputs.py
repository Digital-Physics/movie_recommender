from torch import nn
import pandas as pd
import faiss

# Below, we outline steps for doing a two ReLu neural net tower ranking predictor with user and item embeddings as feature inputs.
# This could be seen as a little overkill since we are concatenating in additional features that may be sufficient on their own for getting a good embedding at the top of the ReLu towers.

###### Offline training
# Step 1a: parse & clean data

# Step 1b: train, validation, test split 
# Note: we do train, validation, test split before we do feature engineering because we don't want any minor data leakage from our holdout data (e.g. normalizing a feature by dividing by an average that includes test data)
# It may not be a serious data leak problem to split the data after feature engineering, and it may make applying these transformations easier, but it may not be a best practice.

# Step 1c: feature engineering 
# We start off using historical data to train out model, but we'd like to fold in our new data and retrain our model as we get new user engagement with our recommendations
# Note: we should log raw user data (w/ timestamps) so our features can evolve over versions of the recommender model, if needed.
# we don't want to lose information by assuming all we need to capture is the features we are currently using.

# Step 2a: Get user and item embeddings through collaborative filtering or some other method.
# Here we'll import our collaborative filtering embeddings.
user_embeddings = faiss.read_index("../embeddings/user_embeddings.index")
item_embeddings = faiss.read_index("../embeddings/item_embeddings.index")

# Step 2b (done in two_tower_predictor_with_embedding_features.py): train two tower ReLu Neural Net with a dot product at the last layer. 
# We should have a good mix of both positive and negative examples, or in our case numbers between 0 and 10.
# Ground truth will be a number between 0 and 10. Coefficient 1, 2, 3, 4 will be used to weight click, like, comment, and share. (Loss will MSE)
# We could make the ground truth user-specific (e.g. if user engages by likes, overweight likes.), but we'll just use the same composite ground truth across all users:
# ground_truth_item_i_user_j = 1*click_i_by_user_j + 2*like_i_by_user_j +3*comment_i_by_user_j + 4*share_i_by_user_j.
# Note: we may also want to explore boosting architectures such as XGBoost, CatBoost, or LGBM after our Neural Net approach. 

# Step 2c: save the trained model in the model registry. Here we'll just import the trained model from our directory.

###### Serving in Production
# Step 3: Evaluate the performance with A/B tests. Use the new model on a subset of users before evaluating whether it should replace our current default model in Production.
# Some evaluation metrics could include Area under the ROCurve. Greater 70% or greater than current model could be threshold.
# If we are using a regression metric and not a binary classification, we could look at MSE, R**2

# Step 4a: FAISS retrieval to find the 250 closest vector candidates (we only have ~10k movies, but let's pretend there were billions). These might be stored in a vector DB.
# We can use an approximate nearest neighbor algorithm like FAISS

# Step 4b: Rank the 250 candidate items for user j by doing 250 forward passes on our trained two tower model (inference) 
# This could be done in parallel... map(two_tower, [zip(left_tower_user_features * 250, right_tower_item_features)])
# We can use the user and item embeddings we obtained through collaborative filtering or some other method as feature inputs
# We can concatenate additional temporal context features for the user. We could also add some metadata features to the item side. These would be interpretable features we understand as opposed to the embedding vectors.

# Step 4c: Post-process the rankings. Select for diversity of recommendations, filter content you don't want, etc. which is essentially re-ranking

# Step 5: Log raw user interaction data. Collect new data for the next collaborative filtering matrix factorization (for embedding features) and the next two tower re-training.

# Step 6: Periodically retrain two models referenced in step 5 for new users, items, and interactions.

# Thoughts to consider: 
# Serving. How quickly can inference be done? Can we score all the candidate items in parallel?
# What is our online learning approach for retraining?
# Perhaps steps 5 and 6 make us rethink whether we really want to retrain to get new embeddings all the time, especially if our items (or users) are coming in quickly.

# Additional Links:

# HOPSWORKS seems to be an alternative to MLFlow
# ML Pipeline = 3 independent pipelines
# Feature Pipeline (raw data to features) + 
# Training Pipeline (train model w/ features & labels) + 
# Inference Pipeline (predictions w/ models and features)
# Does Training Pipeline get new feedback from the post-Inference Pipeline?
https://www.youtube.com/watch?v=s20w8nKCK2o
https://www.youtube.com/watch?v=9vBRjGgdyTY

# Also, let's consider this outline from a mock interview or two
https://www.youtube.com/watch?v=7_E4wnZGJKo
https://www.youtube.com/watch?v=4mG7morAasw


