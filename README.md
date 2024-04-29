This repository has recommender system code related to collaborative learning.

The recommender.ipynb comes from the following Recommender System tutorial & repository:
https://www.youtube.com/watch?v=XfAe-HLysOM
https://github.com/topspinj/tmls-2020-recommender-workshop

This notebook has some additional notes, modifications, and code I added as I went through. There was also some supplemental ratings that were added to the ratings file dataset.

This collaborative filtering (item-to-item and user-to-item) model would probably fall under batch recommendation systems, where the model is trained offline and recommended items are saved to a database. There is still more to explore here, including distance metric differences, gradient descent vs SVD for matrix factorization, post-processing code (e.g. filter movies that were seen & predicted below 4), composite utility values (e.g. (e.g. utility_matrix[user][j] = hours_watched_movie_j + rating_movie_j)), normalizing (e.g. rating_user_i/average_rating_from_user_i)), etc.

We are also going to look at realtime recommendations using a retrieval & ranking architecture. In this architecture, we go from billions of possible candidates, 
down to hundreds, and then run each of those through a predictor for ranking purposes. The click and user interaction data can then be used to update the user's recent history/context. We can have either a one or two ReLu tower neural net. The potential benefit of a two tower approach is that you are bringing the enriched user and item embeddings together in a dot product (followed by a sigmoid for squashing between 0 and 1 if click/no-click is your ground truth, or maybe no sigmoid if your ground truth is a composite utility metric that can range from say 0 to 15) only at the end, not at the base of the neural net ReLu tower. In this kind of model, we may do something like collaborative learning matrix factorization to get the static user and item embeddings, and then add the context (recent likes, clicks, comments, forwards, save, friend activity/likes, etc.) into the user tower. 

Post processing (e.g. filter for inappropriate content, already watched videos, etc.) can be used in either instance.

![architecture image](./img/retrieval_ranking.png)
![two tower](./img/two_tower.png)

https://www.youtube.com/watch?v=9vBRjGgdyTY



