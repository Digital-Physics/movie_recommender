# Fake data
documents = ['document1', 'document2', 'document3', 'document4', 'document5']
# how do we determine the ground truth of relevancy for recommender systems? 
# click-through? but what about unsurfaced documents that were never seen? how do we know our False Negatives?
relevant_documents = ['document1', 'document2', 'document3'] 

# Retrieval system output
retrieved_documents = ['document1', 'document4']

# Calculating precision (TP retrieved/ (TP retrieved + FP retrieved))
relevant_retrieved = [doc for doc in retrieved_documents if doc in relevant_documents]
precision = len(relevant_retrieved) / len(retrieved_documents) if len(retrieved_documents) > 0 else 0

# Calculating recall (TP retrieved/ (TP + FN (not retrieved but were relevant)))
relevant_retrieved = [doc for doc in relevant_documents if doc in retrieved_documents]
recall = len(relevant_retrieved) / len(relevant_documents) if len(relevant_documents) > 0 else 0

print("Precision:", precision)
print("One of two documents that were retrieved were actually relevant. (TP/(TP + FP))")
print()
print("Recall:", recall)
print("Only one of the three relevant documents were retrieved. (TP/(TP + FN))")
print("How do we identify FN in recommender systems?")