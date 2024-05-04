import torch
from torch import nn


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
    
    def forward(self, item_features, user_features, context_features):
        item_output = self.item_tower(item_features)
        user_output = self.user_tower(torch.cat((user_features, context_features), dim=1))
        dot_product = torch.sum(item_output * user_output, dim=1, keepdim=True)
        return torch.sigmoid(dot_product)

# Example usage
# we have two separate Bag of Words for user search history and item text
# should we use a more sophisticated embedding for text?
bag_of_words_embedding_length = 32 # reduced from vocab length to 32 using PCA or t-SNE 
last_item_click = 64

# to do: think of more/better features. 
# to do: consider whether we need item context features for the item... so both the item and user are a rep of their entire history and their their relevance to now/recent vs the past
# to do: change this code into df.columns once we have some fake data
item_feature_size = len(["post_age", "likes_per_hour", "user_id"]) + bag_of_words_embedding_length
user_feature_size = len(["user_age", "geo_location", "logins_per_week"]) 
context_feature_size = last_item_click + bag_of_words_embedding_length
model = TwoTowerModel(item_feature_size, user_feature_size, context_feature_size)

# Example inputs
item_features = torch.randn(1, item_feature_size)
user_features = torch.randn(1, user_feature_size)
context_features = torch.randn(1, context_feature_size)

# Forward pass
output = model(item_features, user_features, context_features)
print(output)