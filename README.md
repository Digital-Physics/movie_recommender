This repository has recommender system code related to collaborative learning and retrieval & ranking.

The recommender.ipynb comes from the following Recommender System tutorial on retrieval and ranking:
https://www.youtube.com/watch?v=XfAe-HLysOM
https://github.com/topspinj/tmls-2020-recommender-workshop

This notebook has some additional notes, modifications, and code I added as I went through. There was also some supplemental ratings that were added to the ratings file dataset.

This collaborative filtering (item-to-item and user-to-item) model would probably fall under batch recommendation systems, where the model is trained offline and recommended items are saved to a database. There is still more to explore here, including distance metric differences, gradient descent vs SVD for matrix factorization, post-processing code (e.g. filter movies that were seen & predicted below 4), composite utility values (e.g. utility_matrix[user_i][movie_j] = hours_watched_movie_j + rating_movie_j + 2 * likes_movie_j + 5 * shared_movie_j + 2 * comment[i][j]/sum(comments[i])), normalizing (e.g. rating_user_i/average_rating_from_user_i)), ways for handling missing values in sparse vectors (e.g. imputing average, ignoring, treating as 0, treating as infinity), etc.

![SVD or Gradient Descent Utility Matrix factorization](./img/utility_matrix_factorization.png)

We are also going to look at realtime recommendations using a retrieval & ranking architecture. In this approach, we go from billions of possible candidates
down to hundreds in an approximate k nearest neighbor (lower time complexity than actual KNN) retrieval step, and then run each of those through a predictor for ranking purposes. The context/user session info/recent history like recent views and user interaction data can then be used and updated to make evolving recommendations that have some local, temporal context. We can have either a one or two ReLu tower neural net. I think the potential benefit of a two tower approach (over a one ReLu tower neural net) is that you are bringing the enriched user and item embeddings together in a dot product (followed by a sigmoid for squashing between 0 and 1 if click/no-click is your ground truth, or maybe no sigmoid or a ReLu if your ground truth is a composite utility metric that can range from say 0 to 15) only at the end, not at the base of the neural net ReLu tower. In this kind of model, we may do something like collaborative learning matrix factorization to get the static user and item embeddings, and then add the context (recent likes, clicks, comments, forwards, save, friend activity/likes, etc.) into the user tower. 

Post-processing (e.g. filter for inappropriate content, already watched videos, etc.) can be used in either instance.

![architecture image](./img/retrieval_ranking.png)
![two tower](./img/two_tower.png)

https://www.youtube.com/watch?v=9vBRjGgdyTY



