# Step 1: data cleaning

# Step 1b: train, validation, test split 
# Note: we do train, validation, test split before we do feature engineering because we don't want any minor data leakage from our holdout data (e.g. dividing by an average)
# It may not be a serious data leak problem to split the data after feature engineering (say in the case of normalization based on all the data, including test data), but it's perhaps a good practice

# Step 1c: feature engineering
# we start off using historical data to train out model, but we can fold new data and retrain our model as we get new user engagement with our recommendations
# good practice note: we should log raw user data (w/ timestamps) so our features can evolve over versions of the recommender model, if needed.
# we don't want to lose information by assuming all we need to capture is the features we are currently using.

# Step 2: train two tower ReLu Neural Net with a dot product at the last layer. We should have a good mix of both positive and negative examples, or in our case numbers between 0 and 10.
# Ground truth will be a number between 0 and 10. Coefficient 1, 2, 3, 4 will be used to weight click, like, comment, and share. (Loss will MSE)
# We could make the ground truth user-specific (e.g. if user engages by likes, overweight likes.), but we'll just use the same composite ground truth across all users:
# ground_truth_item_i_user_j = 1*click_i_by_user_j + 2*like_i_by_user_j +3*comment_i_by_user_j + 4*share_i_by_user_j.
# Note: we may also want to explore boosting architectures such as XGBoost, CatBoost, or LGBM 

# Step 2b: save the trained model in the model registry (in our case we'll just save it in memory)

# Step 3: FAISS retrieval to find the 250 closest vector candidates (we only have ~10k movies, but let's pretend there were billions). These might be stored in a vector DB.

# Step 4: Rank the 250 candidate items for user j by doing 250 forward passes on our two tower model (inference on the trained model) (could be done in parallel map(two_tower, [ (left_tower_user_features) (right_tower_item_i_features)])
# We can use the static user embedding and static item embeddings we obtained through collaborative learning or some other method
# We can also add, and some features that show

# Step 5: Post-process the rankings. Select for diversity of recommendations, filter content, etc. which is essentially re-ranking

# Step 6: Log raw user interaction data. Collectollect new data for next training of the

# use trained two tower predictor (on Toy Story) on say 250 movies
# create fake user data?
https://www.youtube.com/watch?v=9vBRjGgdyTY

# HOPSWORKS seems to be an alternative to MLFlow
# ML Pipeline = 3 independent pipelines
# Feature Pipeline (raw data to features) + 
# Training Pipeline (train model w/ features & labels) + 
# Inference Pipeline (predictions w/ models and features)
# Does Training Pipeline get new feedback from the post-Inference Pipeline?
https://www.youtube.com/watch?v=s20w8nKCK2o

# Also, let's consider this outline from a mock interview or two
https://www.youtube.com/watch?v=7_E4wnZGJKo
https://www.youtube.com/watch?v=4mG7morAasw


