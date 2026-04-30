from src.DataProcessing import tokenizer
from collections import defaultdict
import math


class LanguageModel:
    def __init__(self, docs, relevance, model='unigram', lambda_=0.5):
        self.model = model
        self.lambda_ = lambda_
        self.doc_ids = docs['doc_id'].tolist()

        # Preserve token order for bigram; duplicates count for unigram TF
        self.doc_tokens = [tokenizer(doc) for doc in docs['text']]

        # Per-document unigram TF and length
        self.doc_tf = []
        self.doc_lengths = []
        for tokens in self.doc_tokens:
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self.doc_tf.append(tf)
            self.doc_lengths.append(len(tokens))

        # Collection unigram TF and total length
        self.collection_tf = defaultdict(int)
        self.collection_length = 0
        for tf, length in zip(self.doc_tf, self.doc_lengths):
            for term, count in tf.items():
                self.collection_tf[term] += count
            self.collection_length += length

        # Bigram counts per document and across the collection
        if model == 'bigram':
            self.doc_bigram_tf = []
            self.collection_bigram_tf = defaultdict(int)
            for tokens in self.doc_tokens:
                bg = defaultdict(int)
                for i in range(1, len(tokens)):
                    pair = (tokens[i - 1], tokens[i])
                    bg[pair] += 1
                    self.collection_bigram_tf[pair] += 1
                self.doc_bigram_tf.append(bg)

        self.ap_scores = {}
        self.sorted_ap = []

        self.relevant_docs = (
            relevance[relevance['relevance'] == 1]
            .groupby('query_id')['doc_id']
            .apply(set)
            .to_dict()
        )

    def _p_smooth(self, term, doc_idx):
        L_d = self.doc_lengths[doc_idx]
        if L_d == 0:
            return 0.0
        p_doc = self.doc_tf[doc_idx].get(term, 0) / L_d
        p_col = self.collection_tf.get(term, 0) / self.collection_length
        return self.lambda_ * p_doc + (1 - self.lambda_) * p_col

    def _unigram_score(self, query_tokens, doc_idx):
        score = 0.0
        for term in query_tokens:
            p = self._p_smooth(term, doc_idx)
            if p <= 0:
                return float('-inf')
            score += math.log(p)
        return score

    def _bigram_score(self, query_tokens, doc_idx):
        if not query_tokens:
            return 0.0

        p_first = self._p_smooth(query_tokens[0], doc_idx)
        if p_first <= 0:
            return float('-inf')
        score = math.log(p_first)

        tf_d = self.doc_tf[doc_idx]
        bg_tf_d = self.doc_bigram_tf[doc_idx]

        for i in range(1, len(query_tokens)):
            prev, curr = query_tokens[i - 1], query_tokens[i]
            prev_count = tf_d.get(prev, 0)
            p_bigram_doc = bg_tf_d.get((prev, curr), 0) / prev_count if prev_count > 0 else 0.0
            p_unigram_col = self.collection_tf.get(curr, 0) / self.collection_length
            p = self.lambda_ * p_bigram_doc + (1 - self.lambda_) * p_unigram_col
            if p <= 0:
                return float('-inf')
            score += math.log(p)

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
            q_tokens = tokenizer(q_text)

            if self.model == 'unigram':
                scores = [self._unigram_score(q_tokens, i) for i in range(len(self.doc_ids))]
            else:
                scores = [self._bigram_score(q_tokens, i) for i in range(len(self.doc_ids))]

            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            self.ap_scores[q_id] = self._compute_ap(ranked_indices, self.relevant_docs.get(q_id, set()))

        self.sorted_ap = sorted(self.ap_scores.items(), key=lambda x: x[1], reverse=True)
        return self.sorted_ap
