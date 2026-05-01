# Global Precomputed Values
from src.DataProcessing import DataProcessing

# Vector Space Models
from src.vsm_search import VSMSearch

# Probability Based Models
from src.bm25_search import BM25Search
from src.bim_search import BIMSearch

# Language Models
from src.language_model import LanguageModel

def main():
    p = DataProcessing() # default datapath is "./dataset"
    
    # print(p.docs)       # gets documents in collection
    # print(p.qrels)      # gets relevance in collection
    # print(p.queries)    # gets queries in collection
    # print(p.tfidf)      # gets tfidf matrix using documents

    bm25 = BM25Search(p.docs, p.qrels)
    vsm = VSMSearch(p.docs, p.qrels, p.vectorizer, p.tfidf)
    bim = BIMSearch(p.docs, p.qrels)
    unigram = LanguageModel(docs=p.docs, relevance=p.qrels, model='unigram', lambda_=0.3)
    bigram = LanguageModel(docs=p.docs, relevance=p.qrels, model='bigram', lambda_=0.3)


    ranked_docs = bm25.search(p.queries)
    sorted_bm25 = bm25.getAPScores()
    
    print("====================================================")
    print("BM25")
    print(ranked_docs)    
    for q, ap in sorted_bm25:
        print(f'{q}: {ap:.4f}')
    

    ranked_vsm = vsm.search(p.queries)
    sorted_vsm = vsm.getAPScores()

    print("====================================================")
    print("VSM")   
    print(ranked_vsm)
    for q, ap in sorted_vsm:
        print(f'{q}: {ap:.4f}')


    ranked_bim = bim.search(p.queries)
    sorted_bim = bim.getAPScores()

    print("====================================================")
    print("BIM")   
    print(ranked_bim)
    for q, ap in sorted_bim:
        print(f'{q}: {ap:.4f}')


    ranked_uni = unigram.search(p.queries)
    sorted_unigram = unigram.getAPScores()

    print("====================================================")
    print("LM Unigram")   
    print(ranked_uni)
    for q, ap in sorted_unigram:
        print(f'{q}: {ap:.4f}')


    ranked_bi = bigram.search(p.queries)
    sorted_bigram = bigram.getAPScores()

    print("====================================================")
    print("LM Bigram")   
    print(ranked_bi)
    for q, ap in sorted_bigram:
        print(f'{q}: {ap:.4f}')
    
if __name__ == "__main__":
    main()
