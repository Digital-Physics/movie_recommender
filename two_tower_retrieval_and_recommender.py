from torch import nn
import pandas as pd
import faiss
from two_tower_predictor import TwoTowerModel  # trained already

###### Serving in Production
# Step 3: Evaluate the performance with A/B tests. Use the new model on a subset of users before evaluating whether it should replace our current default model in Production.

# Step 3a: FAISS retrieval to find the 250 closest vector candidates (we only have ~10k movies, but let's pretend there were billions). These might be stored in a vector DB.
# We can use an approximate nearest neighbor algorithm like FAISS
model = TwoTowerModel(item_feature_size, user_feature_size, context_feature_size)
predictions = model(item_features, user_features, context_features)

# Step 3b: Rank the 250 candidate items for user j by doing 250 forward passes on our trained two tower model (inference) 
# This could be done in parallel... map(two_tower, [zip(left_tower_user_features * 250, right_tower_item_features)])

# Step 3c: Post-process the rankings. Select for diversity of recommendations, filter content you don't want, etc. which is essentially re-ranking

# Step 4: Update evaluation metrics with new data
# We could include Area under the ROCurve. Greater 70% or greater than current model could be threshold.
# If we are using a regression metric and not a binary classification, we could look at MSE, R**2

# Step 5: Log raw user interaction data. Collect data for the next two tower re-training. Raw data will allow are features to evolve if we want to change something in the future.
# Note: we should log raw user data (w/ timestamps) so our features can evolve over versions of the recommender model, if needed.
# We start off using historical data to train our model, but we'd like to fold in our new data and retrain our model as we get new user engagement with our recommendations
# we don't want to lose information by assuming all we need to capture is the features we are currently using.

# Step 6: Periodically retrain two tower model

# Thoughts to consider: 
# Serving in production.
# Latency?: How quickly can retrieval and inference be done? Can we score all the candidate items in parallel?
# Scalability?: Can we scale servers as we get more items and users? Do our algorithms scale?
# Retraining?: What is our online learning approach for retraining?

