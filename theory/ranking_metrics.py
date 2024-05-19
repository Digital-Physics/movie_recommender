import numpy as np

# Example ground truth relevance scores for 3 users (they each clicked on 2 or 3 of the 5 videos)
ground_truth = {
    # like videos 1, 2, 3, 4, 5
    'user1': [1, 0, 1, 0, 0], # liked: 1,3; rr = 1/2 (first vid liked in ranking was 2nd movie, movie 1), P@k=3: 2/3
    'user2': [1, 1, 0, 0, 1], # liked: 1,2,5; rr = 1/1, P@k=3: 2/3
    'user3': [0, 1, 0, 1, 1]  # liked: 2,4,5; rr = 1/1, P@k=3: 2/3
}

# Example predicted ranking of videos for each user
predicted_ranking = {
    'user1': [2, 1, 3, 5, 4],
    'user2': [1, 2, 3, 4, 5],
    'user3': [5, 4, 3, 2, 1]
}

def calculate_mrr(ground_truth, predicted_ranking):
    """calculates Mean Reciprocal Rank (MRR)
    simple metric to calc and understand: 
    the mean across users of 1 / ranking_of_first_relevant_item"""
    # (1/2 + 1/1 + 1/1)/3 = 5/6 = 0.8333
    mrr_sum = 0
    for user, gt in ground_truth.items():
        rr = next((1/(i+1) for i, pred_rank in enumerate(predicted_ranking[user]) if gt[pred_rank - 1] == 1), 0)
        mrr_sum += rr
    return mrr_sum / len(ground_truth)

def calculate_ndcg(ground_truth, predicted_ranking, k):
    """calculate Normalized Discounted Cumulative Gain (NDCG) at k.
    Note: like in finance, the higher rankings, like the distant Cash Flows will get discounted and be worth less than the low rankings/less distant CFs"""
    ndcg_sum = 0
    for user, gt in ground_truth.items():
        dcg = sum([gt[pred_rank-1] / np.log2(i + 2) for i, pred_rank in enumerate(predicted_ranking[user][:k])])
        idcg = sum([(2**gt[pred_rank-1] - 1) / np.log2(i + 2) for i, pred_rank in enumerate(sorted(gt, reverse=True)[:k])])
        ndcg_sum += dcg / idcg
    return ndcg_sum / len(ground_truth)

def calculate_precision_at_k(ground_truth, predicted_ranking, k):
    """calculate Precision at k (P@k), where k is the ranking cut-off for calculating precision."""
    # k = 3 (not to be confused with n=3 users): (2/3 + 2/3 + 2/3)/3 = 2/3 = 0.6666 (note the denominator is n and doesn't have to match k)
    precision_sum = 0
    for user, gt in ground_truth.items():
        precision = sum([gt[pred_rank-1] for pred_rank in predicted_ranking[user][:k]]) / k
        precision_sum += precision
    return precision_sum / len(ground_truth)

def calculate_map(ground_truth, predicted_ranking):
    """calculate Mean Average Precision (MAP)
    the mean across the users of (the (arithmetic) average p@k across k where k is relevant)
    note: dividing by the TP relevance count for each user means users who engaged w/ 2 items instead of 3 aren't necessarily hurt
    """
    # ((0 + 1/2 + 2/3 + 0 + 0)/2 + (1/1 + 2/2 + 0 + 0 + 3/5)/3 + (1/1 + 2/2 + 0 + 3/4 + 0)/3) / 3 = 0.7888
    # sum terms: P@k * (1 if k was rel/engaged else 0)
    # did Alice say multiply terms by mistake?
    map_sum = 0
    for user, gt in ground_truth.items():
        num_correct = 0
        precision_sum = 0
        for i, pred_rank in enumerate(predicted_ranking[user]):
            if gt[pred_rank-1] == 1:
                num_correct += 1
                precision_sum += num_correct / (i + 1)
        map_sum += precision_sum / sum(gt)  # dividing by the positive ground truth engagement count means users who engaged w/ 2 instead of 3 posts aren't hurt
    return map_sum / len(ground_truth)

# Calculating metrics
mrr = calculate_mrr(ground_truth, predicted_ranking)
ndcg = calculate_ndcg(ground_truth, predicted_ranking, 3)  # NDCG at 3
precision_at_3 = calculate_precision_at_k(ground_truth, predicted_ranking, 3)
map_score = calculate_map(ground_truth, predicted_ranking)

print(f'MRR: {mrr:.2f}')
print(f'NDCG@3: {ndcg:.2f}')
print(f'Precision@3: {precision_at_3:.2f}')
print(f'MAP: {map_score:.2f}')