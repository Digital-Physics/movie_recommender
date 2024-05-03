import torch
from torch import nn


class TwoTowerModel(nn.Module):
    def __init__(self, item_embedding_size, user_embedding_size, context_feature_size):
        """"this two tower model imports embeddings, but we may not have item embeddings (or user embeddings) from a collaborative filtering kind of process if new items don't have an embedding yet.
        also, we are assuming the output is a binary classifier (click/no-click) but we could have a regression model with a score (click = 1, comment = 2, like = 3, score = 4)."""
        # super() is used to call the constructor of the parent class of TwoTowerModel, which is nn.Module. 
        # This is necessary because TwoTowerModel is inheriting from nn.Module, and you need to initialize the nn.Module part of TwoTowerModel before adding your custom functionality.
        # When you inherit from a parent class, Python creates a new class that includes all the attributes and methods of the parent class. 
        # However, to properly initialize the parent class's attributes and set up its internal state, you need to explicitly call the parent class's constructor
        super(TwoTowerModel, self).__init__()
        self.item_tower = nn.Sequential(
            nn.Linear(item_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.user_tower = nn.Sequential(
            nn.Linear(user_embedding_size + context_feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
    
    def forward(self, item_embedding, user_embedding, context_features):
        item_output = self.item_tower(item_embedding)
        user_output = self.user_tower(torch.cat((user_embedding, context_features), dim=1))
        dot_product = torch.sum(item_output * user_output, dim=1, keepdim=True)
        return torch.sigmoid(dot_product)

# Example usage
item_embedding_size = 20 # this is our item embedding feature input size, not our final item embedding size of 64
user_embedding_size = 20 # this is our user embedding feature input size, not our final user embedding size of 64
context_feature_size = 5
model = TwoTowerModel(item_embedding_size, user_embedding_size, context_feature_size)

# Example inputs
item_embedding = torch.randn(1, item_embedding_size)
user_embedding = torch.randn(1, user_embedding_size)
context_features = torch.randn(1, context_feature_size)

# Forward pass
output = model(item_embedding, user_embedding, context_features)
print(output)