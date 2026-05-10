import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.DataProcessing import DataProcessing
from src.vsm_search import VSMSearch
from src.bm25_search import BM25Search
from src.bim_search import BIMSearch
from src.language_model import LanguageModel

from eval.metrics import get_p_at_k, get_all_p_at_k, get_per_query_p_at_k, get_pr_curve, get_runtime, get_pareto_frontier, get_combined_score
import eval.plots as plotter

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print("Loading dataset...")
    p = DataProcessing(DATA_DIR='dataset',
                       QRELS_DIR='qrels_secret_team_01.json',
                       Q_DIR='secret_queries_team_01.json')
    
    models = {
        "BM25": BM25Search(p.docs, p.qrels),
        "VSM": VSMSearch(p.docs, p.qrels, p.vectorizer, p.tfidf),
        "BIM": BIMSearch(p.docs, p.qrels),
        "Unigram LM": LanguageModel(docs=p.docs, relevance=p.qrels, model='unigram', lambda_=0.3),
        "Bigram LM": LanguageModel(docs=p.docs, relevance=p.qrels, model='bigram', lambda_=0.3)
    }
    
    map_scores = {}
    p5_scores = {}
    all_ap_scores = {}
    pr_curves = {}
    patk_curves = {}
    per_query_p10 = {}
    runtimes = {}

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    for name, model in models.items():
        runtimes[name] = get_runtime(p, model)
        sorted_ap = model.getAPScores()

        ap_dict = {q_id: ap for q_id, ap in sorted_ap}
        all_ap_scores[name] = ap_dict

        map_scores[name] = sum(ap_dict.values()) / len(ap_dict) if ap_dict else 0.0
        p5_scores[name] = get_p_at_k(model, k=5)

        recall_levels, avg_precisions = get_pr_curve(model)
        pr_curves[name] = (recall_levels, avg_precisions)
        patk_curves[name] = get_all_p_at_k(model, max_k=10)
        per_query_p10[name] = get_per_query_p_at_k(model, k=10)

    names = list(map_scores.keys())
    ranks = get_combined_score(names, map_scores, runtimes)
    pareto_names = get_pareto_frontier(names, map_scores, runtimes)

    print("Generating figures...")
    plotter.plot_map_comparison(names, list(map_scores.values()), colors, figures_dir)
    plotter.plot_per_query_ap(all_ap_scores, colors, figures_dir)
    plotter.plot_pr_curve(pr_curves, colors, figures_dir)
    plotter.plot_patk_curve(names, patk_curves, colors, figures_dir)
    plotter.plot_metrics_table(names, map_scores, p5_scores, figures_dir)
    plotter.plot_runtime(names, runtimes, colors, figures_dir)
    plotter.plot_runtime_map_table(names, map_scores, runtimes, ranks, figures_dir)
    plotter.plot_pareto_frontier(names, map_scores, runtimes, pareto_names, colors, figures_dir)

    for idx, name in enumerate(names):
        plotter.plot_blind_per_query_ap(name, all_ap_scores[name], map_scores[name], colors[idx], figures_dir)
        plotter.plot_blind_per_query_p_at_k(name, per_query_p10[name], 10, colors[1], figures_dir)

    print(f"Figures saved to /eval/figures")

if __name__ == "__main__":
    main()