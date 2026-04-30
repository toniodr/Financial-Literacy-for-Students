import pandas as pd
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

class ProcessesData:
    def __init__(self, DATA_DIR="dataset"):
        """Creates document, query, and relevance dataframe.

        Args:
            DATA_DIR (str, optional): Path to dataset directory. Defaults to "dataset".
        """
        data_dir = Path(DATA_DIR)
        with open(data_dir / "documents.json", encoding='utf-8') as f:
            self.docs = pd.DataFrame(json.load(f))
        
        with open(data_dir / "qrels.json", encoding='utf-8') as f:
            self.qrels = pd.DataFrame(json.load(f))

        with open(data_dir / "queries.json", encoding='utf-8') as f:
            self.queries = pd.DataFrame(json.load(f))

        # Calculate tfidf cause we prob need it for whatever ranking we decide to do
        tfidfvectorizer = TfidfVectorizer(
            stop_words='english'
        )

        self.tfidf = tfidfvectorizer.fit_transform(self.docs['text'])
        
