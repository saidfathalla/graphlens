#!/usr/bin/env python3
import os
import re
import yaml
import logging
import requests
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rdflib
from rdflib.util import guess_format
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Suppress noisy semantic parsing logs from terminal output
logging.getLogger("rdflib.term").setLevel(logging.ERROR)

class GraphLensAgnosticPipeline:
    def __init__(self, profile_path_or_url, training_data_path="data/training_data.csv", output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Ingest and map YAML dynamic attributes (from local file or remote URL)
        self.load_profile_configuration(profile_path_or_url)
        
        # Core GraphLens Heuristic Setup
        self.layout_clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
        self.vectorizer = CountVectorizer()
        self._bootstrap_layout_classifier(training_data_path)
        
    def load_profile_configuration(self, path_or_url):
        """Ingests profile.yml definitions from a local path or remote URL string."""
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            print(f"[*] Fetching remote profile configuration: {path_or_url}")
            try:
                response = requests.get(path_or_url, timeout=30)
                response.raise_for_status()
                profile = yaml.safe_load(response.text)
            except Exception as e:
                raise RuntimeError(f"Failed to fetch remote profile configuration: {e}")
        else:
            if not os.path.exists(path_or_url):
                raise FileNotFoundError(f"Configuration profile not found at: {path_or_url}")
            with open(path_or_url, "r", encoding="utf-8") as f:
                profile = yaml.safe_load(f)
            
        self.domain_ns = profile.get("domain_namespace", "")
        self.target_class = profile.get("target_class", "")
        
        attrs = profile.get("attributes", {})
        
        # Extract Independent Metrics (x1, x2, etc.)
        self.independent_metrics = {item['identifier']: item['predicate'] for item in attrs.get('independent_metrics', [])}
        
        # Extract Derived Ratios (y)
        self.derived_ratios = {item['identifier']: item['predicate'] for item in attrs.get('derived_ratios', [])}
        
        # Extract Nominal Categories (c1, c2)
        self.nominal_categories = {item['identifier']: item['predicate'] for item in attrs.get('nominal_categories', [])}
        
        # Primary key shortcuts for analytical loops
        self.volume_key = "x1" if "x1" in self.independent_metrics else list(self.independent_metrics.keys())[0]
        self.success_key = "x2" if "x2" in self.independent_metrics else None
        self.rate_key = "y" if "y" in self.derived_ratios else list(self.derived_ratios.keys())[0]
        self.cat_key = "c1" if "c1" in self.nominal_categories else None
        self.geo_key = "c2" if "c2" in self.nominal_categories else None
        
        print(f"[*] Profile loaded successfully. Target class tracking: {self.target_class}")

    def _bootstrap_layout_classifier(self, path):
        """Trains the internal heuristic mapping model for layout classification."""
        if os.path.exists(path):
            df = pd.read_csv(path)
            X = df["columns_text"].tolist()
            y = df["target_chart"].tolist()
        else:
            X = [
                "year x1 x2 y", 
                "c1 y variation", 
                "c2 entries cumulative",
                "cluster_label volume profile"
            ]
            y = ["Line", "Scatter", "Bar", "ClusterPanel"]
            
        X_vec = self.vectorizer.fit_transform(X)
        self.layout_clf.fit(X_vec, y)
        print("[*] GraphLens ML Layout Engine compiled successfully.")

    def clean_and_normalize_payload(self, df):
        """Performs dynamic layout parsing according to active YAML profile specifications."""
        if df.empty:
            return df
            
        print("[*] Parsing timeline sequences and cleaning semantic namespaces...")
        
        if 'date' in df.columns:
            df['year'] = df['date'].apply(
                lambda x: int(re.search(r'\d{4}', str(x)).group()) 
                if pd.notna(x) and re.search(r'\d{4}', str(x)) else np.nan
            )
        elif 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        else:
            df['year'] = np.nan

        for col in df.columns:
            if col not in ['date', 'year']:
                df[col] = df[col].apply(
                    lambda x: str(x).strip().split('/')[-1].split('#')[-1] if pd.notna(x) else np.nan
                )
        
        numeric_targets = list(self.independent_metrics.keys()) + list(self.derived_ratios.keys())
        for col in numeric_targets:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if self.volume_key in df.columns and self.success_key in df.columns and self.rate_key in df.columns:
            mask_acc = df[self.success_key].isna() & df[self.rate_key].notna() & (df[self.volume_key] > 0)
            df.loc[mask_acc, self.success_key] = (df.loc[mask_acc, self.volume_key] * df.loc[mask_acc, self.rate_key]).round()
            
            mask_rate = df[self.rate_key].isna() & (df[self.volume_key] > 0) & df[self.success_key].notna()
            df.loc[mask_rate, self.rate_key] = df.loc[mask_rate, self.success_key] / df.loc[mask_rate, self.volume_key]
            
            if df[self.rate_key].max() > 1.0:
                df[self.rate_key] = df[self.rate_key] / 100.0
                
        df = df.dropna(subset=['year', self.volume_key]).copy()
        df['year'] = df['year'].astype(int)
        
        return df

    def execute_advanced_analytics(self, df):
        print("[*] Running profile-aligned discovery engine calculations...")
        
        clustering_data = df[[self.volume_key, self.rate_key]].dropna()
        if len(clustering_data) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(clustering_data)
            df.loc[clustering_data.index, 'cluster_label'] = labels
            
            cluster_means = df.groupby('cluster_label')[self.rate_key].mean()
            sorted_clusters = cluster_means.sort_values().index
            cluster_mapping = {
                sorted_clusters[0]: "High Selectivity Target Band",
                sorted_clusters[1]: "Standard Distribution Band",
                sorted_clusters[2]: "High Density Acceptance Band"
            }
            df['agnostic_profile'] = df['cluster_label'].map(cluster_mapping).fillna("General Ingested Cluster")
        else:
            df['agnostic_profile'] = "General Ingested Cluster"

        yearly_aggregates = df.groupby('year').agg({self.volume_key: 'sum'}).reset_index()
        if len(yearly_aggregates) > 1:
            X_reg = yearly_aggregates[['year']].values
            y_reg = yearly_aggregates[self.volume_key].values
            reg_model = LinearRegression().fit(X_reg, y_reg)
            df_meta = {
                "growth_slope": reg_model.coef_[0],
                "r_squared": reg_model.score(X_reg, y_reg)
            }
            print(f"[+] Empirical growth trend computed: {df_meta['growth_slope']:.2f} [{self.volume_key}] units/year (R² = {df_meta['r_squared']:.2f})")
        else:
            df_meta = {"growth_slope": 0.0, "r_squared": 0.0}

        return df, yearly_aggregates, df_meta

    def generate_publication_plots(self, df, yearly_df, meta):
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        plt.rcParams.update({'font.family': 'serif', 'savefig.dpi': 300, 'figure.autolayout': True})

        print("\n[*] Generating graphical assets outposts aligned to target configuration...\n" + "="*80)

        # FIGURE 1: K-Means Clustering
        if 'agnostic_profile' in df.columns and len(df['agnostic_profile'].unique()) > 1:
            fig1, ax1 = plt.subplots(figsize=(9, 6))
            sns.scatterplot(data=df, x=self.volume_key, y=self.rate_key, hue='agnostic_profile', palette='Set1', s=110, alpha=0.8, edgecolor='black', ax=ax1)
            sns.regplot(data=df, x=self.volume_key, y=self.rate_key, scatter=False, color='gray', line_kws={'linestyle': '--', 'linewidth': 1.5}, ax=ax1)
            ax1.set_title(f"Figure 1: K-Means Clustering Matrix mapping {self.volume_key} vs {self.rate_key}", fontweight='bold', pad=12)
            ax1.set_xlabel(f"Normalized Volume Scale ({self.volume_key})")
            ax1.set_ylabel(f"Calculated Metric Value ({self.rate_key})")
            if df[self.volume_key].max() > 500: ax1.set_xscale('log')
            plt.savefig(os.path.join(self.output_dir, "fig1_cluster_regression.png"))
            plt.close(fig1)

        # FIGURE 2: Longitudinal Distribution
        if self.cat_key and self.cat_key in df.columns and len(df[self.cat_key].dropna().unique()) > 1:
            field_temporal = df.groupby(['year', self.cat_key])[self.volume_key].sum().unstack().fillna(0)
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            field_temporal.plot(kind='area', stacked=True, cmap='tab20', alpha=0.85, ax=ax2)
            ax2.set_title(f"Figure 2: Longitudinal Distribution of {self.volume_key} Across Categories", fontweight='bold', pad=12)
            plt.savefig(os.path.join(self.output_dir, "fig2_field_temporal_growth.png"), bbox_inches='tight')
            plt.close(fig2)

        # FIGURE 3: Geographic Distribution
        if self.geo_key and self.geo_key in df.columns:
            # Changed from ['event'].nunique() to .size() to avoid KeyError
            geo_dist = df.groupby(self.geo_key).size().sort_values(ascending=False).head(12)
            fig3, ax3 = plt.subplots(figsize=(9, 5.5))
            sns.barplot(x=geo_dist.values, y=geo_dist.index, palette='Blues_r', edgecolor='black', ax=ax3)
            ax3.set_title("Figure 3: Geographic Concentration Profile", fontweight='bold', pad=12)
            plt.savefig(os.path.join(self.output_dir, "fig3_geographic_distribution.png"))
            plt.close(fig3)

        # FIGURE 4: Correlation Heatmap
        target_metrics = [c for c in ['year', self.volume_key, self.success_key, self.rate_key] if c in df.columns and c is not None]
        if len(target_metrics) > 1:
            fig4, ax4 = plt.subplots(figsize=(7.5, 6))
            corr_matrix = df[target_metrics].corr(method='pearson')
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.7, cbar=True, ax=ax4)
            ax4.set_title("Figure 4: Core Metric Interaction Alignment Matrix Heatmap", fontweight='bold', pad=12)
            plt.savefig(os.path.join(self.output_dir, "fig4_correlation_matrix.png"))
            plt.close(fig4)

        # FIGURE 5: Selectivity Variances
        if self.cat_key and self.rate_key in df.columns:
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x=self.rate_key, y=self.cat_key, palette='vlag', ax=ax5)
            ax5.set_title("Figure 5: Selectivity Variance Across Categories", fontweight='bold', pad=12)
            plt.savefig(os.path.join(self.output_dir, "fig5_field_selectivity_boxplot.png"))
            plt.close(fig5)

        # FIGURE 6: Top Venues Benchmark
        if 'acronym' in df.columns and self.volume_key in df.columns:
            top_venues = df.groupby('acronym')[self.volume_key].sum().sort_values(ascending=False).head(10)
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_venues.values, y=top_venues.index, palette='flare', ax=ax6)
            ax6.set_title("Figure 6: Top 10 High-Impact Venues", fontweight='bold', pad=12)
            plt.savefig(os.path.join(self.output_dir, "fig6_top_venues_benchmark.png"))
            plt.close(fig6)

        # FIGURE 7: Density Volume Distribution
        if 'agnostic_profile' in df.columns and self.volume_key in df.columns:
            fig7, ax7 = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=df, x='agnostic_profile', y=self.volume_key, palette='Pastel1', ax=ax7)
            ax7.set_yscale('log')
            ax7.set_title("Figure 7: Density Volume by Taxonomy Cluster", fontweight='bold', pad=12)
            plt.savefig(os.path.join(self.output_dir, "fig7_cluster_density_violin.png"))
            plt.close(fig7)

    def execute_pipeline(self, query_str, rdf_url=None, rdf_file=None):
        graph = rdflib.Graph()
        detected_format = None

        if rdf_url:
            print(f"[*] Connecting to live semantic data channel: {rdf_url}")
            try:
                response = requests.get(rdf_url, timeout=30)
                if response.status_code == 404:
                    print(f"\n[!] Fatal Ingestion Error: The requested URL returned a '404 Not Found' status.")
                    return
                response.raise_for_status()
                
                # Content-Type sniffing and fallback format guesser mapping
                detected_format = guess_format(rdf_url)
                if not detected_format:
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'turtle' in content_type or 'ttl' in content_type: detected_format = 'turtle'
                    elif 'rdf+xml' in content_type or 'xml' in content_type: detected_format = 'xml'
                    elif 'json-ld' in content_type: detected_format = 'json-ld'
                    elif 'ntriples' in content_type: detected_format = 'nt'
                
                if not detected_format:
                    detected_format = 'turtle'  # Safe internal fallback default
                    
                graph.parse(data=response.text, format=detected_format)
            except Exception as network_err:
                print(f"[!] Target access failure over networking logic: {network_err}")
                return
        elif rdf_file:
            print(f"[*] Accessing local storage resource asset: {rdf_file}")
            if not os.path.exists(rdf_file):
                print(f"[!] Storage reference missing: {rdf_file}")
                return
            detected_format = guess_format(rdf_file) or 'turtle'
            graph.parse(source=rdf_file, format=detected_format)

        print(f"[+] Graph initialized into engine model space. Context size: {len(graph)} triples.")
        
        try:
            query_job = graph.query(query_str)
        except Exception as query_error:
            print(f"[!] SPARQL Runtime Fault: {query_error}")
            return

        columns = [str(var) for var in query_job.vars]
        rows = [[val.toPython() if hasattr(val, 'toPython') else str(val) if val is not None else None for val in row] for row in query_job]
            
        raw_df = pd.DataFrame(rows, columns=columns)
        if raw_df.empty:
            print("[!] Target alignment extraction yielded an empty dataframe.")
            return

        processed_df = self.clean_and_normalize_payload(raw_df)
        analyzed_df, yearly_df, metadata = self.execute_advanced_analytics(processed_df)
        self.generate_publication_plots(analyzed_df, yearly_df, metadata)
        
        csv_path = os.path.join(self.output_dir, "features_matrix.csv")
        analyzed_df.to_csv(csv_path, index=False)
        print(f"\n[+++] Profile operation finished! Assets written to: '{self.output_dir}/'\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphLens: Configurable Remote Semantic Engine Pipeline")
    parser.add_argument("-q", "--query", required=True, help="Path or HTTP URL to execution target SPARQL file query")
    parser.add_argument("-p", "--profile", required=True, help="Path or HTTP URL to profile layout configuration YAML asset")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--url", help="Remote streaming target URL data connection string")
    group.add_argument("-f", "--file", help="Local storage file path data source track")
    
    args = parser.parse_args()
        
    # Dynamically fetch the SPARQL Query if hosted remotely
    if args.query.startswith("http://") or args.query.startswith("https://"):
        print(f"[*] Fetching remote SPARQL query: {args.query}")
        res = requests.get(args.query, timeout=30)
        res.raise_for_status()
        loaded_sparql_query = res.text
    else:
        if not os.path.exists(args.query):
            print(f"[!] Fatal Error: Query file '{args.query}' not found.")
            exit(1)
        with open(args.query, "r", encoding="utf-8") as f:
            loaded_sparql_query = f.read()
    
    pipeline = GraphLensAgnosticPipeline(profile_path_or_url=args.profile)
    pipeline.execute_pipeline(
        query_str=loaded_sparql_query, 
        rdf_url=args.url, 
        rdf_file=args.file
    )
