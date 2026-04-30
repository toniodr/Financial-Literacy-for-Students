from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pandas as pd
import json
import re

def tokenizer(text):
    """Global Tokenize Method for bm25_search, bim_search, etc. Appears in both index and query process.

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    text = re.sub(r'[^\w\s]', '', text.lower())
    return [t for t in text.split() if t not in ENGLISH_STOP_WORDS] 

class DataProcessing:
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
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf = self.vectorizer.fit_transform(self.docs['text'])