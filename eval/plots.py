import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_map_comparison(names, scores, colors, save_dir):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, scores, color=colors)
    plt.ylabel('Mean Average Precision (MAP)')
    plt.title('Overall Model Performance Comparison')
    plt.ylim(0, max(scores) * 1.15 if scores else 1.0)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
    plt.savefig(os.path.join(save_dir, "map_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_query_ap(all_ap_scores, colors, save_dir):
    df_ap = pd.DataFrame(all_ap_scores).sort_index()
    plt.figure(figsize=(10, 5))
    
    for idx, column in enumerate(df_ap.columns):
        plt.plot(df_ap.index.astype(str), df_ap[column], marker='o', label=column, color=colors[idx])
        
    plt.xlabel('Query ID')
    plt.ylabel('Average Precision (AP)')
    plt.title('Per-Query Performance Across Models')
    plt.legend(title="Search Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    
    plt.savefig(os.path.join(save_dir, "per_query_ap.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curve(pr_curves, colors, save_dir):
    plt.figure(figsize=(8, 6))
    for idx, (name, (recalls, precisions)) in enumerate(pr_curves.items()):
        plt.plot(recalls, precisions, marker='s', label=name, color=colors[idx], linewidth=2, markersize=5)
        
    plt.xlabel('Recall')
    plt.ylabel('Interpolated Precision')
    plt.title('11-Point Interpolated Precision-Recall Curve')
    plt.legend(title="Search Models")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    
    plt.savefig(os.path.join(save_dir, "pr_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_patk_curve(names, patk_curves, colors, save_dir):
    plt.figure(figsize=(8, 6))
    for idx, name in enumerate(names):
        k_data = patk_curves[name]
        k_values = list(k_data.keys())
        p_values = list(k_data.values())
        plt.plot(k_values, p_values, marker='^', label=name, color=colors[idx], linewidth=2, markersize=6)
        
    plt.xlabel('Rank Position (K)')
    plt.ylabel('Average Precision at K (P@K)')
    plt.title('Precision Decay Across Top 10 Search Results')
    plt.xticks(range(1, 11))
    plt.legend(title="Search Models")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0.0, 1.05)
    
    plt.savefig(os.path.join(save_dir, "patk_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_table(names, map_scores, p5_scores, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off') 
    
    col_labels = ['Retrieval Model', 'MAP Score', 'P@5 Score']
    sorted_names = sorted(names, key=lambda n: map_scores[n], reverse=True)
    table_data = [[name, f"{map_scores[name]:.4f}", f"{p5_scores[name]:.4f}"] for name in sorted_names]
    
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8) 
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#EAEAF2')
            
    plt.title('Final Evaluation Metrics', fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, "metrics_table.png"), dpi=300, bbox_inches='tight')
    plt.close()