#!/usr/bin/env python3
"""Clustering baselines (GMM, Ward), stability and OLS diagnostics for a GraphLens run.

Run from the repository root, e.g.
  python3 experiments/evaluate_robustness.py -q queries/eventskg-query.rq -f data/eventskg.ttl \\
      -p profiles/eventskg-profile.yaml -o results/robustness
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score, 
    adjusted_rand_score
)
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from graphlens import GraphLensAgnosticPipeline  # noqa: E402

def load_sparql_query(query_input):
    """Loads query text whether given a file path, URL, or raw SPARQL string."""
    if os.path.isfile(query_input):
        with open(query_input, "r", encoding="utf-8") as f:
            return f.read()
    elif query_input.startswith(("http://", "https://")):
        response = requests.get(query_input, timeout=30)
        response.raise_for_status()
        return response.text
    return query_input

def run_robust_evaluation(query_input, profile, output_dir="output", url=None, file=None,
                          endpoint=None, training="data/training_data.csv"):
    print("[1/3] Running GraphLens pipeline to extract normalized feature matrix...")
    
    sparql_text = load_sparql_query(query_input)

    pipeline = GraphLensAgnosticPipeline(
        profile_path_or_url=profile,
        training_data_path=training,
        output_dir=output_dir,
        seed=42
    )
    
    # Executing using loaded query string
    pipeline.execute_pipeline(query_str=sparql_text, rdf_url=url, rdf_file=file, endpoint=endpoint)

    features_path = os.path.join(output_dir, "features_matrix.csv")
    yearly_path = os.path.join(output_dir, "yearly_aggregates.csv")
    
    df = pd.read_csv(features_path)
    yearly_df = pd.read_csv(yearly_path)
    
    x1, y = pipeline.k_vol, pipeline.k_rate
    report_lines = ["="*60, " GRAPHLENS ROBUSTNESS EVALUATION REPORT", "="*60, ""]

    print("[2/3] Computing clustering baselines and stability metrics...")
    if y in df.columns:
        sub = df[[x1, y]].dropna()
        if pipeline.cl.get("log_volume", False):
            sub[x1] = np.log10(sub[x1].clip(lower=1))
        
        Z = StandardScaler().fit_transform(sub)
        k = pipeline.report.get("clustering", {}).get("k", 2)
        
        models = {
            "K-Means (GraphLens Default)": KMeans(n_clusters=k, random_state=42, n_init=10),
            "Gaussian Mixture (GMM)": GaussianMixture(n_components=k, random_state=42),
            "Agglomerative (Ward)": AgglomerativeClustering(n_clusters=k)
        }
        
        report_lines.extend([f"--- 1. CLUSTERING BASELINES (K={k}) ---", 
                             "Comparing default K-Means against density and hierarchical baselines.\n"])
        
        labels_dict = {}
        for name, model in models.items():
            labels = model.fit_predict(Z) if name == "Gaussian Mixture (GMM)" else model.fit(Z).labels_
            labels_dict[name] = labels
            
            s = silhouette_score(Z, labels)
            db = davies_bouldin_score(Z, labels)
            ch = calinski_harabasz_score(Z, labels)
            
            report_lines.append(f"{name}:")
            report_lines.append(f"  - Silhouette (Higher is better): {s:.3f}")
            report_lines.append(f"  - Davies-Bouldin (Lower is better): {db:.3f}")
            report_lines.append(f"  - Calinski-Harabasz (Higher is better): {ch:.1f}\n")

        # (a) Initialisation stability: same data, different random starts. Shows that
        #     K-means is insensitive to its starting centroids; it does NOT test whether
        #     the clusters survive changes in the data.
        n_iters = 50
        aris = []
        base_labels = labels_dict["K-Means (GraphLens Default)"]
        for i in range(n_iters):
            km_seed = KMeans(n_clusters=k, random_state=i, n_init=10).fit(Z)
            aris.append(adjusted_rand_score(base_labels, km_seed.labels_))

        # (b) Bootstrap stability: computed by GraphLens itself (resample the rows with
        #     replacement, refit, compare with the full-data clustering), from
        #     cluster_validation.csv. This is the figure to report as cluster stability.
        boot = None
        val_path = os.path.join(output_dir, "cluster_validation.csv")
        if os.path.exists(val_path):
            val = pd.read_csv(val_path)
            row = val[val["k"] == k]
            if len(row):
                boot = (float(row["ari_stability_mean"].iloc[0]), float(row["ari_stability_sd"].iloc[0]))
        n_boot = pipeline.cl.get("stability_runs")

        report_lines.extend([
            "--- 2. CLUSTERING STABILITY (K-Means) ---",
            f"(a) Initialisation stability, {n_iters} random starts on the same data:",
            f"  - Mean Adjusted Rand Index (ARI): {np.mean(aris):.3f} +/- {np.std(aris):.3f}",
        ])
        if boot:
            report_lines.extend([
                f"(b) Bootstrap stability, {n_boot} resamples of the data (GraphLens):",
                f"  - Mean Adjusted Rand Index (ARI): {boot[0]:.3f} +/- {boot[1]:.3f}",
            ])
            verdict = boot[0]
        else:
            report_lines.append("(b) Bootstrap stability: cluster_validation.csv not found.")
            verdict = None
        if verdict is None:
            report_lines.append("  - Conclusion: only initialisation stability available.\n")
        elif verdict > 0.9:
            report_lines.append("  - Conclusion: clusters are stable under resampling of the data.\n")
        else:
            report_lines.append("  - Conclusion: clusters change noticeably when the data are resampled.\n")
    else:
        report_lines.append("Clustering skipped: Target rate variable not found.\n")

    print("[3/3] Computing time-series regression diagnostics...")
    if len(yearly_df) > 2:
        report_lines.extend(["--- 3. LONGITUDINAL REGRESSION DIAGNOSTICS ---"])
        X_time = sm.add_constant(yearly_df['year'])
        y_total = yearly_df['total']
        
        ols_model = sm.OLS(y_total, X_time).fit()
        residuals = ols_model.resid
        
        dw = durbin_watson(residuals)
        shapiro_stat, shapiro_p = shapiro(residuals)
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_time)
        
        report_lines.append("Validating statistical assumptions of the OLS model:")
        report_lines.append(f"  - R-squared: {ols_model.rsquared:.3f}")
        report_lines.append(f"  - Autocorrelation (Durbin-Watson): {dw:.2f} (Ideal: ~2.0)")
        report_lines.append(f"  - Normality of Residuals (Shapiro-Wilk p-value): {shapiro_p:.3e} " 
                            f"{'(Pass: Normal)' if shapiro_p > 0.05 else '(Fail: Non-normal)'}")
        report_lines.append(f"  - Homoscedasticity (Breusch-Pagan p-value): {bp_p:.3e} "
                            f"{'(Pass: Constant variance)' if bp_p > 0.05 else '(Fail: Heteroscedastic)'}\n")
    else:
        report_lines.append("Regression diagnostics skipped: Insufficient yearly data points.\n")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    os.makedirs(output_dir, exist_ok=True)
    report_file_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_file_path, "w") as f:
        f.write(report_text)
    print(f"\n[+] Full evaluation report saved to: {report_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphLens Robustness Evaluator")
    parser.add_argument("-q", "--query", required=True, help="SPARQL file or https URL")
    parser.add_argument("-p", "--profile", required=True, help="profile.yaml path")
    parser.add_argument("-o", "--output", default="output")
    parser.add_argument("-t", "--training", default="data/training_data.csv", help="layout training CSV")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("-u", "--url", help="RDF file URL (downloaded and parsed in memory)")
    g.add_argument("-f", "--file", help="Local RDF file")
    g.add_argument("-e", "--endpoint", help="SPARQL endpoint URL (query runs remotely, paged)")

    args = parser.parse_args()
    run_robust_evaluation(args.query, args.profile, args.output, args.url, args.file,
                          endpoint=args.endpoint, training=args.training)
