# Global Precomputed Values
from src.DataProcessing import DataProcessing

# Vector Space Models
from src.vsm_search import VSMSearch

# Probability Based Models
from src.bm25_search import BM25Search
from src.bim_search import BIMSearch

def main():
    p = DataProcessing() # default datapath is "./dataset"
    
    # print(p.docs)       # gets documents in collection
    # print(p.qrels)      # gets relevance in collection
    # print(p.queries)    # gets queries in collection
    # print(p.tfidf)      # gets tfidf matrix using documents

    bm25 = BM25Search(p.docs, p.qrels)
    vsm = VSMSearch(p.docs, p.qrels, p.vectorizer, p.tfidf)
    bim = BIMSearch(p.docs, p.qrels)

    sorted_bm25 = bm25.getAPScores(p.queries)
    print("====================================================")
    print("BM25 Sorted :", end='\n\n')
    for q, ap in sorted_bm25:
        print(f'{q}: {ap:.4f}')
    
    sorted_vsm = vsm.getAPScores(p.queries)
    print("====================================================")
    print("VSM Sorted :", end='\n\n')
    for q, ap in sorted_vsm:
        print(f'{q}: {ap:.4f}')
    
    sorted_bim = bim.getAPScores(p.queries)
    print("====================================================")
    print("BIM Sorted :", end='\n\n')
    for q, ap in sorted_bim:
        print(f'{q}: {ap:.4f}')
        
if __name__ == "__main__":
    main()
