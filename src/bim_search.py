from ProcessesData import tokenizer
from collections import defaultdict
import math
import re

class BIMSearch:
    def __init__(self, docs, relevance):
        self.doc_ids = docs['doc_id'].tolist()
        self.N = len(self.doc_ids)

        # Binary representation: each doc is a set of terms (presence only)
        self.doc_terms = [set(tokenizer(doc)) for doc in docs['text']]

        # n_i: number of documents containing each term
        self.doc_freq = defaultdict(int)
        for terms in self.doc_terms:
            for term in terms:
                self.doc_freq[term] += 1

        self.ap_scores = {}
        self.sorted_ap = []

        self.relevant_docs = (
            relevance[relevance['relevance'] == 1]
            .groupby('query_id')['doc_id']
            .apply(set)
            .to_dict()
        )

    def _bim_score(self, query_terms, doc_term_set):
        # Sum IDF-like weights for terms present in both query and document
        # No relevance feedback (r=0, R=0): log[(N - n_i + 0.5) / (n_i + 0.5)]
        score = 0.0
        for term in query_terms:
            if term in doc_term_set:
                n_i = self.doc_freq.get(term, 0)
                score += math.log((self.N - n_i + 0.5) / (n_i + 0.5))
        return score

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
            query_terms = set(tokenizer(q_text))

            scores = [self._bim_score(query_terms, doc_terms) for doc_terms in self.doc_terms]
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            self.ap_scores[q_id] = self._compute_ap(ranked_indices, self.relevant_docs.get(q_id, set()))

        self.sorted_ap = sorted(self.ap_scores.items(), key=lambda x: x[1], reverse=True)
        return self.sorted_ap
