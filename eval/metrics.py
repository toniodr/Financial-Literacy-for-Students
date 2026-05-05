import numpy as np

def get_p_at_k(model, k=5):
    """Calculate average Precision at K."""
    p_scores = []
    for _, row in model.ranked_docs.iterrows():
        q_id = row['query_id']
        ranked_indices = row['ranked_indices'][:k]
        rel_docs = model.relevant_docs.get(q_id, set())
        
        if not rel_docs:
            continue
            
        tp = sum(1 for idx in ranked_indices if model.doc_ids[idx] in rel_docs)
        p_scores.append(tp / k)
        
    return sum(p_scores) / len(p_scores) if p_scores else 0.0

def get_all_p_at_k(model, max_k=10):
    """calculate average Precision at K for every K from 1 up to max_k."""
    k_scores = {k: [] for k in range(1, max_k + 1)}

    for _, row in model.ranked_docs.iterrows():
        q_id = row['query_id']
        ranked_indices = row['ranked_indices']
        rel_docs = model.relevant_docs.get(q_id, set())

        if not rel_docs:
            continue

        for k in range(1, max_k + 1):
            top_k_indices = ranked_indices[:k]
            tp = sum(1 for idx in top_k_indices if model.doc_ids[idx] in rel_docs)
            k_scores[k].append(tp / k)

    avg_k_scores = {}
    for k, scores in k_scores.items():
        avg_k_scores[k] = sum(scores) / len(scores) if scores else 0.0
        
    return avg_k_scores

def get_pr_curve(model):
    """calculate the standard 11-point interpolated Precision-Recall curve."""
    recall_levels = np.arange(0.0, 1.1, 0.1)
    sum_precisions = np.zeros(11)
    valid_queries = 0

    for _, row in model.ranked_docs.iterrows():
        q_id = row['query_id']
        ranked_indices = row['ranked_indices']
        rel_docs = model.relevant_docs.get(q_id, set())

        if not rel_docs:
            continue

        total_rel = len(rel_docs)
        tp = 0
        precisions = []
        recalls = []

        for rank, idx in enumerate(ranked_indices, start=1):
            if model.doc_ids[idx] in rel_docs:
                tp += 1
            precisions.append(tp / rank)
            recalls.append(tp / total_rel)

        interp_p = np.zeros(11)
        for i, r in enumerate(recall_levels):
            valid_p = [p for p, rec in zip(precisions, recalls) if rec >= r]
            interp_p[i] = max(valid_p) if valid_p else 0.0

        sum_precisions += interp_p
        valid_queries += 1

    if valid_queries == 0:
        return recall_levels, np.zeros(11)
    
    return recall_levels, sum_precisions / valid_queries