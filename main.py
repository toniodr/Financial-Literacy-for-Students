from src.ProcessesData import ProcessesData
from src.bm25_search import BM25Search
from src.vsm_search import VSMSearch

def main():
    p = ProcessesData() # default datapath is "./dataset"
    bm25 = BM25Search(p.docs, p.qrels)
    
    # print(p.docs)       # gets documents in collection
    # print(p.qrels)      # gets relevance in collection
    # print(p.queries)    # gets queries in collection
    # print(p.tfidf)      # gets tfidf matrix using documents

    vsm = VSMSearch(p.docs, p.qrels, p.vectorizer, p.tfidf)

    sorted_bm25 = bm25.getAPScores(p.queries)
    print("====================================================")
    print("BM25 Sorted :")
    for q, ap in sorted_bm25:
        print(f'{q}: {ap:.4f}')

    sorted_vsm = vsm.getAPScores(p.queries)
    print("====================================================")
    print("VSM Sorted :")
    for q, ap in sorted_vsm:
        print(f'{q}: {ap:.4f}')
        
if __name__ == "__main__":
    main()
