from ProcessesData import ProcessesData


def main():
    p = ProcessesData() # default datapath is "./dataset"
    print(p.docs)       # gets documents in collection
    print(p.qrels)      # gets relevance in collection
    print(p.queries)    # gets queries in collection
    print(p.tfidf)      # gets tfidf matrix using documents


if __name__ == "__main__":
    main()