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

def plot_runtime_map_table(names, map_scores, runtimes, ranks, save_dir):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')

    col_labels = ['Retrieval Model', 'MAP Score', 'Runtime (s)']
    sorted_names = sorted(names, key=lambda n: map_scores[n], reverse=True)
    fastest = min(runtimes.values()) if runtimes else 0.0
    table_data = [
        [name, f"{map_scores[name]:.4f}", f"{runtimes[name]:.4f}"]
        for name in sorted_names
    ]

    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#EAEAF2')
        elif col == 2 and table_data[row - 1][2] == f"{fastest:.4f}":
            cell.set_text_props(weight='bold')

    plt.title('MAP vs. Runtime', fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, "runtime_map_table.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pareto_frontier(names, map_scores, runtimes, pareto_names, colors, save_dir):
    pareto_pts = sorted((runtimes[n], map_scores[n]) for n in pareto_names)

    plt.figure(figsize=(8, 6))

    for idx, name in enumerate(names):
        r, m = runtimes[name], map_scores[name]
        on_frontier = name in pareto_names
        plt.scatter(r, m, color=colors[idx], s=140,
                    edgecolors='black' if on_frontier else 'gray',
                    linewidths=2 if on_frontier else 0.8, zorder=3)
        plt.annotate(name, (r, m), xytext=(8, 8), textcoords='offset points',
                     fontsize=10, fontweight='bold' if on_frontier else 'normal')

    if len(pareto_pts) >= 2:
        xs = [p[0] for p in pareto_pts]
        ys = [p[1] for p in pareto_pts]
        plt.plot(xs, ys, 'k--', alpha=0.6, linewidth=1.5, zorder=2,
                 label=f'Pareto frontier ({len(pareto_pts)} engines)')
        plt.legend(loc='best')

    plt.xscale('log')
    plt.xlabel('Runtime (seconds, log scale)')
    plt.ylabel('Mean Average Precision (MAP)')
    plt.title('MAP vs. Runtime — Pareto Frontier')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(os.path.join(save_dir, "pareto_frontier.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_runtime(names, runtimes, colors, save_dir):
    values = [runtimes[name] for name in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=colors)
    plt.ylabel('Search Runtime (seconds)')
    plt.title('Per-Model Search Runtime')
    plt.ylim(0, max(values) * 1.15 if values else 1.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + max(values) * 0.01,
                 f'{yval:.3f}s', ha='center', va='bottom', fontweight='bold')

    plt.savefig(os.path.join(save_dir, "runtime.png"), dpi=300, bbox_inches='tight')
    plt.close()
