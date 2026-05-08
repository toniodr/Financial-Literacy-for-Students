import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.DataProcessing import DataProcessing, tokenizer


def main():
    print("Loading dataset...")
    p = DataProcessing()
    
    num_docs = len(p.docs)
    num_queries = len(p.queries)
    
    print("Tokenizing documents to calculate length...")
    doc_lengths = p.docs['text'].apply(lambda x: len(tokenizer(x)))
    avg_doc_length = doc_lengths.mean()
    
    print("Tokenizing queries to calculate length...")
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

    os.makedirs("stats", exist_ok=True)

    # write report to a text file
    txt_filename = "stats/dataset_stats.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Statistics have been saved to: {txt_filename}")
    
    table_data = [
        ["Total Documents", f"{num_docs:,}"],
        ["Total Queries", f"{num_queries:,}"],
        ["Average Document Length", f"{avg_doc_length:.2f} tokens"],
        ["Average Query Length", f"{avg_query_length:.2f} tokens"],
        ["Total Relevant Judgments", f"{total_rel_judgments:,}"],
        ["Avg Relevant Docs per Query", f"{avg_rel_per_query:.2f}"],
        ["Max Relevant Docs for a Query", f"{max_rel_per_query}"],
        ["Min Relevant Docs for a Query", f"{min_rel_per_query}"]
    ]
    columns = ["Metric", "Value"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')

    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='left', loc='center', colWidths=[0.6, 0.4])

    # style table 
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    # Add header styling and alternate row colors for readability
    for i in range(len(table_data) + 1):
        for j in range(len(columns)):
            cell = table[i, j]
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e0e0e0') 
            else:
                if i % 2 == 0:
                    cell.set_facecolor('#f9f9f9') 

    plt.title("Dataset Description and Analysis", size=14, pad=10)
    plt.tight_layout()

    # Save the figure to the stats folder
    png_filename = "stats/dataset_stats_table.png"
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Table figure has been saved to: {png_filename}")

if __name__ == "__main__":
    main()