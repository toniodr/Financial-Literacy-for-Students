import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class BM25Search:
    def __init__(self, docs, relevance):
        self.doc_ids = docs['doc_id'].tolist()
        self.tokenized_docs = [self.preprocess(doc) for doc in docs['text']]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.ap_scores = {}
        self.sorted_ap = {}

        self.relevant_docs = (
            relevance[relevance['relevance'] == 1]
            .groupby('query_id')['doc_id']
            .apply(set)
            .to_dict()
        )

    def preprocess(self, text):
        tokens = text.lower().split()
        return [t for t in tokens if t not in ENGLISH_STOP_WORDS]

    def _compute_ap(self, ranked_indices, relevant_doc_ids):
        total_rel = len(relevant_doc_ids)
        if total_rel == 0:
            return 0.0
        ap_sum, rel_found = 0, 0
        for rank, i in enumerate(ranked_indices, start=1):
            if self.doc_ids[i] in relevant_doc_ids:
                rel_found += 1
                ap_sum += rel_found / rank
        return ap_sum / total_rel

    def getAPScores(self, queries):
        for _, row in queries.iterrows():
            q_id, q_text = row['query_id'], row['text']
            scores = self.bm25.get_scores(self.preprocess(q_text))
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            self.ap_scores[q_id] = self._compute_ap(ranked_indices, self.relevant_docs.get(q_id, set()))

        self.sorted_ap = sorted(self.ap_scores.items(), key=lambda x: x[1], reverse=True)
        return self.sorted_ap