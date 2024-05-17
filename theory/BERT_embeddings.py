from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Description text
description = "Your video description here"

# Tokenize input
inputs = tokenizer(description, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

# Get model output
with torch.no_grad():
    outputs = model(**inputs)

# Extract embeddings
embedding = outputs.last_hidden_state[:, 0, :].numpy()

print(embedding)