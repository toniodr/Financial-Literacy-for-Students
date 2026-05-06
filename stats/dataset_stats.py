import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.DataProcessing import DataProcessing, tokenizer


def main():
    print("Loading dataset...")
    p = DataProcessing()
    
    num_docs = len(p.docs)
    num_queries = len(p.queries)
    
    doc_lengths = p.docs['text'].apply(lambda x: len(tokenizer(x)))
    avg_doc_length = doc_lengths.mean()
    
    query_lengths = p.queries['text'].apply(lambda x: len(tokenizer(x)))
    avg_query_length = query_lengths.mean()
    
    relevant_only = p.qrels[p.qrels['relevance'] == 1]
    rel_per_query = relevant_only.groupby('query_id').size()
    
    avg_rel_per_query = rel_per_query.mean()
    max_rel_per_query = rel_per_query.max()
    min_rel_per_query = rel_per_query.min()
    total_rel_judgments = len(relevant_only)

    report_content = (
        f"{'='*50}\n"
        f"DATASET STATISTICS REPORT\n"
        f"{'='*50}\n"
        f"Total Documents:                 {num_docs:,}\n"
        f"Total Queries:                   {num_queries:,}\n"
        f"{'-'*50}\n"
        f"Average Document Length:         {avg_doc_length:.2f} tokens\n"
        f"Average Query Length:            {avg_query_length:.2f} tokens\n"
        f"{'-'*50}\n"
        f"Total Relevant Judgments:        {total_rel_judgments:,}\n"
        f"Avg Relevant Docs per Query:     {avg_rel_per_query:.2f}\n"
        f"Max Relevant Docs for a Query:   {max_rel_per_query}\n"
        f"Min Relevant Docs for a Query:   {min_rel_per_query}\n"
        f"{'='*50}\n"
    )

    # write report to a text file
    output_filename = "stats/dataset_stats.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"statistics have been saved to: {output_filename}")

if __name__ == "__main__":
    main()