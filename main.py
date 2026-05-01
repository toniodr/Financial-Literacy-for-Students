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
    print("Ranked Indices per Query")
    print(ranked_docs)    
    print("====================================================")
    print("BM25 Sorted :", end='\n\n')
    for q, ap in sorted_bm25:
        print(f'{q}: {ap:.4f}')
    
    
    # sorted_vsm = vsm.getAPScores(p.queries)
    # print("====================================================")
    # print("VSM Sorted :", end='\n\n')
    # for q, ap in sorted_vsm:
    #     print(f'{q}: {ap:.4f}')
    
    # sorted_bim = bim.getAPScores(p.queries)
    # print("====================================================")
    # print("BIM Sorted :", end='\n\n')
    # for q, ap in sorted_bim:
    #     print(f'{q}: {ap:.4f}')
        
    # sorted_unigram = unigram.getAPScores(p.queries)
    # print("====================================================")
    # print("Unigram Sorted :", end='\n\n')
    # for q, ap in sorted_unigram:
    #     print(f'{q}: {ap:.4f}')

    # sorted_bigram = bigram.getAPScores(p.queries)
    # print("====================================================")
    # print("Bigram Sorted :", end='\n\n')
    # for q, ap in sorted_bigram:
    #     print(f'{q}: {ap:.4f}')
    
if __name__ == "__main__":
    main()
