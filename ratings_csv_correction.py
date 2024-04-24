import pandas as pd
from datetime import datetime

ratings = pd.read_csv('./data/ratings.csv')

users = ratings['userId'].unique()

# a = appending
with open('./data/ratings.csv', 'a') as file:
    for user in users:
        rating = f"{user},{200000},{5.0},{int(datetime.now().timestamp())}"
        file.write(rating + '\n')