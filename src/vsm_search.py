from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class VSMSearch:
    def __init__(self, docs, relevance, vectorizer, tfidf_matrix):
        self.doc_ids = docs['doc_id'].tolist()
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.ap_scores = {}
        self.sorted_ap = []
        self.ranked_docs = pd.DataFrame()

        self.relevant_docs = (
            relevance[relevance['relevance'] == 1]
            .groupby('query_id')['doc_id']
            .apply(set)
            .to_dict()
        )

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

    def search(self, queries):
        self.ranked_docs = pd.DataFrame()
        for _, row in queries.iterrows():
            q_id, q_text = row['query_id'], row['text']
            scores = cosine_similarity(self.vectorizer.transform([q_text]), self.tfidf_matrix)[0]
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranked_scores = [scores[i] for i in ranked_indices]
            self.ranked_docs = pd.concat([self.ranked_docs, pd.DataFrame([{'query_id': q_id, 'text': q_text, 'ranked_indices': ranked_indices, 'scores': ranked_scores}])])
        return self.ranked_docs

    def getAPScores(self):
        for _, row in self.ranked_docs.iterrows():
            q_id, ranked_indices = row['query_id'], row['ranked_indices']
            self.ap_scores[q_id] = self._compute_ap(ranked_indices, self.relevant_docs.get(q_id, set()))

        self.sorted_ap = sorted(self.ap_scores.items(), key=lambda x: x[1], reverse=True)
        return self.sorted_ap
