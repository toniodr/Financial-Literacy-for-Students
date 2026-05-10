from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from pathlib import Path
import pandas as pd
import json
import re

stemmer = PorterStemmer()

def tokenizer(text):
    """Global Tokenize Method for bm25_search, bim_search, etc. Appears in both index and query process.

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """

    # surface level normalization 
    text = text.lower();

    # remove punctuation 
    text = re.sub(r'[^\w\s]', '', text)

    # normalize white space to a single space 
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenize and stopping
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS] 

    # stemming
    stemmed_tokens = [stemmer.stem(t) for t in tokens]

    return stemmed_tokens

class DataProcessing:
    def __init__(self, DATA_DIR="dataset", QRELS_DIR='qrels.json', Q_DIR='queries.json'):
        """Creates document, query, and relevance dataframe.

        Args:
            DATA_DIR (str, optional): Path to dataset directory. Defaults to "dataset".
        """
        data_dir = Path(DATA_DIR)

        with open(data_dir / "documents.json", encoding='utf-8') as f:
            self.docs = pd.DataFrame(json.load(f))
        
        with open(data_dir / QRELS_DIR, encoding='utf-8') as f:
            self.qrels = pd.DataFrame(json.load(f))

        with open(data_dir / Q_DIR, encoding='utf-8') as f:
            self.queries = pd.DataFrame(json.load(f))

        # Calculate tfidf cause we prob need it for whatever ranking we decide to do
        # passed the tokenizer function as the tokenizer for tfidf vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenizer, token_pattern=None)
        self.tfidf = self.vectorizer.fit_transform(self.docs['text'])