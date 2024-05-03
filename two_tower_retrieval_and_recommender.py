from torch import nn
import pandas as pd
import faiss

###### Offline training
# Step 1a: parse & clean data

# Step 1b: train, validation, test split 
# Note: we do train, validation, test split before we do feature engineering because we don't want any minor data leakage from our holdout data (e.g. normalizing a feature by dividing by an average that includes test data)
# It may not be a serious data leak problem to split the data after feature engineering, and it may make applying these transformations easier, but it may not be a best practice.

# Step 1c: feature engineering 
# We start off using historical data to train out model, but we'd like to fold in our new data and retrain our model as we get new user engagement with our recommendations
# Note: we should log raw user data (w/ timestamps) so our features can evolve over versions of the recommender model, if needed.
# we don't want to lose information by assuming all we need to capture is the features we are currently using.

# Step 2b (done in two_tower_predictor.py): train two tower ReLu Neural Net with a dot product at the last layer. 
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

# Step 4c: Post-process the rankings. Select for diversity of recommendations, filter content you don't want, etc. which is essentially re-ranking

# Step 5: Log raw user interaction data. Collect new data for the next collaborative filtering matrix factorization (for embedding features) and the next two tower re-training.

# Step 6: Periodically retrain two models referenced in step 5 for new users, items, and interactions.

# Thoughts to consider: 
# Serving in production.
# Latency?: How quickly can retrieval and inference be done? Can we score all the candidate items in parallel?
# Scalability?: Can we scale servers as we get more items and users? Do our algorithms scale?
# Retraining?: What is our online learning approach for retraining?

