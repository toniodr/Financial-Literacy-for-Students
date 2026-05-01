from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.DataProcessing import tokenizer
from rank_bm25 import BM25Okapi
import pandas as pd
import re

class BM25Search:
    def __init__(self, docs, relevance):
        self.doc_ids = docs['doc_id'].tolist()
        self.tokenized_docs = [tokenizer(doc) for doc in docs['text']]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.ap_scores = {}
        self.sorted_ap = []
        self.ranked_docs = pd.DataFrame()
        
        self.relevant_docs: dict = (
            relevance[relevance['relevance'] == 1]
            .groupby('query_id')['doc_id']
            .apply(set)
            .to_dict()
        )

    def _compute_ap(self, ranked_indices: list, relevant_doc_ids: set):
        total_rel = len(relevant_doc_ids)
        if total_rel == 0:
            return 0.0
        ap_sum, rel_found = 0, 0
        for rank, i in enumerate(ranked_indices, start=1):
            if self.doc_ids[i] in relevant_doc_ids:
                rel_found += 1
                ap_sum += rel_found / rank
        return ap_sum / total_rel

    def search(self, queries: pd.DataFrame):
        self.ranked_docs = pd.DataFrame()
        for _, row in queries.iterrows():
            q_id, q_text = row['query_id'], row['text']
            scores = self.bm25.get_scores(tokenizer(q_text))
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            self.ranked_docs = pd.concat([self.ranked_docs, pd.DataFrame([{'query_id': q_id, 'text': q_text, 'ranked_indices': ranked_indices}])])
        return self.ranked_docs
    
    def getAPScores(self):
        for _, row in self.ranked_docs.iterrows():
            q_id, ranked_indices = row['query_id'], row['ranked_indices']
            self.ap_scores[q_id] = self._compute_ap(ranked_indices, self.relevant_docs.get(q_id, set()))

        self.sorted_ap = sorted(self.ap_scores.items(), key=lambda x: x[1], reverse=True)
        return self.sorted_ap 