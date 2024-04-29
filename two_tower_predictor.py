from torch import nn

class TwoTowerModel(nn.Module):
    def __init__(self, item_embedding_size, user_embedding_size, context_feature_size):
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
item_embedding_size = 10
user_embedding_size = 20
context_feature_size = 5
model = TwoTowerModel(item_embedding_size, user_embedding_size, context_feature_size)

# Example inputs
item_embedding = torch.randn(1, item_embedding_size)
user_embedding = torch.randn(1, user_embedding_size)
context_features = torch.randn(1, context_feature_size)

# Forward pass
output = model(item_embedding, user_embedding, context_features)
print(output)