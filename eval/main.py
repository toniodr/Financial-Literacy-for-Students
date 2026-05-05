import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.DataProcessing import DataProcessing
from src.vsm_search import VSMSearch
from src.bm25_search import BM25Search
from src.bim_search import BIMSearch
from src.language_model import LanguageModel

from eval.metrics import get_p_at_k, get_all_p_at_k, get_pr_curve
import eval.plots as plotter

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print("Loading dataset...")
    p = DataProcessing()
    
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
    
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    
    for name, model in models.items():
        model.search(p.queries)
        sorted_ap = model.getAPScores()
        
        ap_dict = {q_id: ap for q_id, ap in sorted_ap}
        all_ap_scores[name] = ap_dict
        
        map_scores[name] = sum(ap_dict.values()) / len(ap_dict) if ap_dict else 0.0
        p5_scores[name] = get_p_at_k(model, k=5)
        
        recall_levels, avg_precisions = get_pr_curve(model)
        pr_curves[name] = (recall_levels, avg_precisions)
        patk_curves[name] = get_all_p_at_k(model, max_k=10)

    names = list(map_scores.keys())

    print("Generating figures...")
    plotter.plot_map_comparison(names, list(map_scores.values()), colors, figures_dir)
    plotter.plot_per_query_ap(all_ap_scores, colors, figures_dir)
    plotter.plot_pr_curve(pr_curves, colors, figures_dir)
    plotter.plot_patk_curve(names, patk_curves, colors, figures_dir)
    plotter.plot_metrics_table(names, map_scores, p5_scores, figures_dir)

    print(f"Figures saved to /eval/figures")

if __name__ == "__main__":
    main()