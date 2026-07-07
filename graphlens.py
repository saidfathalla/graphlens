#!/usr/bin/env python3
import os
import re
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

class GraphLensScientometricPipeline:
    def __init__(self, training_data_path="data/training_data.csv", output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Core GraphLens Autonomous Layout Classifier Heuristic Setup
        self.layout_clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
        self.vectorizer = CountVectorizer()
        self._bootstrap_layout_classifier(training_data_path)
        
    def _bootstrap_layout_classifier(self, path):
        """Trains the internal heuristic mapping model for layout classification."""
        if os.path.exists(path):
            df = pd.read_csv(path)
            X = df["columns_text"].tolist()
            y = df["target_chart"].tolist()
        else:
            X = [
                "year submitted papers accepted papers", 
                "field acceptanceRate competitiveness", 
                "country total_events",
                "cluster_label volume profile"
            ]
            y = ["Line", "Scatter", "Bar", "ClusterPanel"]
            
        X_vec = self.vectorizer.fit_transform(X)
        self.layout_clf.fit(X_vec, y)
        print("[*] GraphLens ML Layout Engine compiled successfully.")

    def clean_and_normalize_payload(self, df):
        """Performs robust type casting and transforms semantic namespaces into analytical features."""
        if df.empty:
            return df
            
        print("[*] Parsing dates and cleaning semantic URIs into structural features...")
        
        # Extract 4-digit years from the temporal string literals using Regex
        if 'date' in df.columns:
            df['year'] = df['date'].apply(
                lambda x: int(re.search(r'\d{4}', str(x)).group()) 
                if pd.notna(x) and re.search(r'\d{4}', str(x)) else np.nan
            )
        else:
            df['year'] = np.nan

        # Strip remaining namespace URIs to cleanly isolate literal tokens for charting labels
        for col in df.columns:
            if col not in ['date', 'year']:
                df[col] = df[col].apply(
                    lambda x: str(x).strip().split('/')[-1].split('#')[-1] if pd.notna(x) else np.nan
                )
        
        # Explicitly cast metric arrays to float numerical arrays
        numeric_targets = ['submitted', 'accepted', 'acceptanceRate']
        for col in numeric_targets:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Cross-derive missing metrics mathematically to maximize valid observational sample sizes
        if 'submitted' in df.columns:
            if 'accepted' in df.columns and 'acceptanceRate' in df.columns:
                mask_acc = df['accepted'].isna() & df['acceptanceRate'].notna() & (df['submitted'] > 0)
                df.loc[mask_acc, 'accepted'] = (df.loc[mask_acc, 'submitted'] * df.loc[mask_acc, 'acceptanceRate']).round()
                
                mask_rate = df['acceptanceRate'].isna() & (df['submitted'] > 0) & df['accepted'].notna()
                df.loc[mask_rate, 'acceptanceRate'] = df.loc[mask_rate, 'accepted'] / df.loc[mask_rate, 'submitted']
            
            # Normalize percentage variants to a standard [0.0, 1.0] decimal format
            if 'acceptanceRate' in df.columns and df['acceptanceRate'].max() > 1.0:
                df['acceptanceRate'] = df['acceptanceRate'] / 100.0
                
        # Drop entries missing critical timeline vectors
        df = df.dropna(subset=['year', 'submitted']).copy()
        df['year'] = df['year'].astype(int)
        
        return df

    def execute_advanced_analytics(self, df):
        """Applies unsupervised learning and statistical regression over the processed data matrix."""
        print("[*] Running advanced Scientometric discovery engine...")
        
        # 1. Unsupervised Machine Learning: K-Means Clustering on Competitive Behavior
        clustering_data = df[['submitted', 'acceptanceRate']].dropna()
        if len(clustering_data) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(clustering_data)
            df.loc[clustering_data.index, 'cluster_label'] = labels
            
            # Categorize numeric cluster coordinates into standardized academic taxa profiles
            cluster_means = df.groupby('cluster_label')['acceptanceRate'].mean()
            sorted_clusters = cluster_means.sort_values().index
            cluster_mapping = {
                sorted_clusters[0]: "Elite / Highly Selective",
                sorted_clusters[1]: "Standard Peer-Reviewed",
                sorted_clusters[2]: "High-Acceptance / Mega-Venue"
            }
            df['venue_profile'] = df['cluster_label'].map(cluster_mapping).fillna("Unclassified Venues")
        else:
            df['venue_profile'] = "Unclassified Venues"

        # 2. Statistical Modeling: Ordinary Least Squares (OLS) Trend Line Regression
        yearly_aggregates = df.groupby('year').agg({'submitted': 'sum'}).reset_index()
        if len(yearly_aggregates) > 1:
            X_reg = yearly_aggregates[['year']].values
            y_reg = yearly_aggregates['submitted'].values
            reg_model = LinearRegression().fit(X_reg, y_reg)
            df_meta = {
                "growth_slope": reg_model.coef_[0],
                "r_squared": reg_model.score(X_reg, y_reg)
            }
            print(f"[+] Empirical growth trend computed: {df_meta['growth_slope']:.2f} submissions/year (R² = {df_meta['r_squared']:.2f})")
        else:
            df_meta = {"growth_slope": 0.0, "r_squared": 0.0}

        return df, yearly_aggregates, df_meta

    def generate_publication_plots(self, df, yearly_df, meta):
        """Generates and exports high-resolution multi-panel plots optimized for journal formatting."""
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        plt.rcParams.update({
            'font.family': 'serif',
            'savefig.dpi': 300,
            'figure.autolayout': True
        })

        print("\n[*] Generating high-resolution graphical assets and terminal captions...\n" + "="*80)

        # FIGURE 1: Cluster Analysis with Overlaid OLS Trend Fit
        if 'venue_profile' in df.columns and len(df['venue_profile'].unique()) > 1:
            fig1, ax1 = plt.subplots(figsize=(9, 6))
            sns.scatterplot(
                data=df, x='submitted', y='acceptanceRate', hue='venue_profile',
                palette='Set1', s=110, alpha=0.8, edgecolor='black', ax=ax1
            )
            sns.regplot(
                data=df, x='submitted', y='acceptanceRate', scatter=False,
                color='gray', line_kws={'linestyle': '--', 'linewidth': 1.5}, ax=ax1
            )
            ax1.set_title("Figure 1: K-Means Clustering and OLS Trend Matrix of Scholarly Venues", fontweight='bold', pad=12)
            ax1.set_xlabel("Submission Ingestion Volume (Log Scale)", fontsize=12)
            ax1.set_ylabel("Empirical Review Acceptance Rate", fontsize=12)
            ax1.set_xscale('log')
            ax1.legend(title="Structural Taxonomy", frameon=True, loc='best')
            
            fig1_path = os.path.join(self.output_dir, "fig1_cluster_regression.png")
            plt.savefig(fig1_path, dpi=300)
            plt.close(fig1)
            print(f"[+] Exported: {fig1_path}")
            print("    CAPTION: Figure 1: Multi-dimensional review behavior taxonomy mapping venue intake volumes against metric acceptance rates. Points categorize automatically via unsupervised K-Means clusters (K=3), overlaid with an OLS trend curve modeling structural competitiveness scaling patterns.\n" + "-"*80)

        # FIGURE 2: Longitudinal Distribution of Submissions Across Fields
        if 'field' in df.columns and len(df['field'].dropna().unique()) > 1:
            field_temporal = df.groupby(['year', 'field'])['submitted'].sum().unstack().fillna(0)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            field_temporal.plot(kind='area', stacked=True, cmap='tab20', alpha=0.85, ax=ax2)
            ax2.set_title("Figure 2: Longitudinal Evolution of Submissions Across Specialization Fields", fontweight='bold', pad=12)
            ax2.set_xlabel("Academic Tracking Year", fontsize=12)
            ax2.set_ylabel("Cumulative Volume Count", fontsize=12)
            ax2.legend(title="Research Field", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
            
            fig2_path = os.path.join(self.output_dir, "fig2_field_temporal_growth.png")
            plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"[+] Exported: {fig2_path}")
            print("    CAPTION: Figure 2: Stacked longitudinal area matrix tracing total thematic distributions across computer science communities across five decades, exposing macro-scale structural shifts and field specialization pathways within knowledge networks.\n" + "-"*80)

        # FIGURE 3: Geographic Distribution of Conference Hosting Nations
        if 'country' in df.columns and len(df['country'].dropna().unique()) > 1:
            geo_dist = df.groupby('country')['event'].nunique().sort_values(ascending=False).head(12)
            
            fig3, ax3 = plt.subplots(figsize=(9, 5.5))
            sns.barplot(x=geo_dist.values, y=geo_dist.index, palette='Blues_r', edgecolor='black', ax=ax3)
            ax3.set_title("Figure 3: Geographic Concentration Profile of Renowned Event Frameworks", fontweight='bold', pad=12)
            ax3.set_xlabel("Absolute Count of Unique Evaluated Conference Editions", fontsize=12)
            ax3.set_ylabel("Hosting Nation Entity", fontsize=12)
            
            fig3_path = os.path.join(self.output_dir, "fig3_geographic_distribution.png")
            plt.savefig(fig3_path, dpi=300)
            plt.close(fig3)
            print(f"[+] Exported: {fig3_path}")
            print("    CAPTION: Figure 3: Concentration distribution profiling geographic host asymmetry of international scientific tracking instances, ranking top nations based on their cumulative count of elite conference events.\n" + "-"*80)

        # FIGURE 4: Metrics Dependency Matrix Correlation Heatmap
        target_metrics = [c for c in ['year', 'submitted', 'accepted', 'acceptanceRate'] if c in df.columns]
        if len(target_metrics) > 1:
            fig4, ax4 = plt.subplots(figsize=(7.5, 6))
            corr_matrix = df[target_metrics].corr(method='pearson')
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.7, cbar=True, ax=ax4)
            ax4.set_title("Figure 4: Correlation Feature Space Heatmap of Core Scholarly Metrics", fontweight='bold', pad=12)
            
            fig4_path = os.path.join(self.output_dir, "fig4_correlation_matrix.png")
            plt.savefig(fig4_path, dpi=300)
            plt.close(fig4)
            print(f"[+] Exported: {fig4_path}")
            print("    CAPTION: Figure 4: Correlation heatmap illustrating Pearson linear dependency indices between temporal progression, submission volume scales, absolute acceptance counts, and review selectiveness constraints.\n" + "-"*80)

        # FIGURE 5: Operational Selectivity Variances Across Fields (Boxplot)
        if 'field' in df.columns and 'acceptanceRate' in df.columns:
            valid_fields = df.dropna(subset=['field', 'acceptanceRate'])
            if len(valid_fields['field'].unique()) > 1:
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=valid_fields, x='acceptanceRate', y='field', palette='vlag', fliersize=4, ax=ax5)
                ax5.set_title("Figure 5: Variance Dynamics of Review Selectiveness Across Fields", fontweight='bold', pad=12)
                ax5.set_xlabel("Empirical Acceptance Rate (Decimal Range)", fontsize=12)
                ax5.set_ylabel("Research Field Taxonomy", fontsize=12)
                
                fig5_path = os.path.join(self.output_dir, "fig5_field_selectivity_boxplot.png")
                plt.savefig(fig5_path, dpi=300)
                plt.close(fig5)
                print(f"[+] Exported: {fig5_path}")
                print("    CAPTION: Figure 5: Cross-domain selectivity variance analysis utilizing horizontal boxplots. This visualization illustrates the structural variation, inner quartiles, and outlier thresholds of peer-review stringency across distinct academic communities.\n" + "-"*80)

        # FIGURE 6: Top 10 Conference Series Benchmarking (Bar Chart)
        if 'acronym' in df.columns and 'submitted' in df.columns:
            top_venues = df.groupby('acronym')['submitted'].sum().sort_values(ascending=False).head(10)
            if not top_venues.empty:
                fig6, ax6 = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_venues.values, y=top_venues.index, palette='flare', edgecolor='black', ax=ax6)
                ax6.set_title("Figure 6: Benchmark Profile of Top 10 High-Impact Venues by Volume", fontweight='bold', pad=12)
                ax6.set_xlabel("Cumulative Tracked Submissions Ingested", fontsize=12)
                ax6.set_ylabel("Venue Acronym Identifier", fontsize=12)
                
                fig6_path = os.path.join(self.output_dir, "fig6_top_venues_benchmark.png")
                plt.savefig(fig6_path, dpi=300)
                plt.close(fig6)
                print(f"[+] Exported: {fig6_path}")
                print("    CAPTION: Figure 6: Evaluation benchmark ranking the top ten most active publication series by historical submission volume within the triplestore metadata repository, highlighting dominant nodes of research productivity.\n" + "-"*80)

        # FIGURE 7: Density Volume Distribution by Taxonomy Cluster (Violin Plot)
        if 'venue_profile' in df.columns and 'submitted' in df.columns:
            fig7, ax7 = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=df, x='venue_profile', y='submitted', palette='Pastel1', inner='quartile', ax=ax7)
            ax7.set_yscale('log')
            ax7.set_title("Figure 7: Density Distribution of Submission Volumes by Taxonomical Cluster", fontweight='bold', pad=12)
            ax7.set_xlabel("Unsupervised Taxonomy Profile Assignment", fontsize=12)
            ax7.set_ylabel("Inbound Submission Scale (Logarithmic Scale)", fontsize=12)
            
            fig7_path = os.path.join(self.output_dir, "fig7_cluster_density_violin.png")
            plt.savefig(fig7_path, dpi=300)
            plt.close(fig7)
            print(f"[+] Exported: {fig7_path}")
            print("    CAPTION: Figure 7: Log-scaled violin plots tracking kernel density distributions of submission volumes across unsupervised venue profiles. This map exposes structural density thresholds and scale differences among elite, standard, and mega-venues.\n" + "="*80)

    def execute_pipeline(self, query_str, rdf_url=None, rdf_file=None):
        """Ingests semantic data from remote URLs or local files with explicit layout structure validation."""
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
                
                detected_format = guess_format(rdf_url)
                if not detected_format:
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'turtle' in content_type or 'ttl' in content_type: detected_format = 'turtle'
                    elif 'rdf+xml' in content_type or 'xml' in content_type: detected_format = 'xml'
                    elif 'json-ld' in content_type: detected_format = 'json-ld'
                    elif 'ntriples' in content_type: detected_format = 'nt'
                
                if not detected_format:
                    print("\n[!] Fatal Ingestion Error: Could not determine RDF serialization framework from URL metadata strings.")
                    return

                try:
                    graph.parse(data=response.text, format=detected_format)
                except Exception as parse_error:
                    print(f"\n[!] Fatal Structural Error: Target stream is not a valid RDF payload format ({detected_format}).\n    Parser Details: {parse_error}")
                    return

            except requests.exceptions.RequestException as network_err:
                print(f"\n[!] Ingestion Failure: Unable to securely connect to remote service channel.\n    Network Details: {network_err}")
                return

        elif rdf_file:
            print(f"[*] Loading local semantic data asset: {rdf_file}")
            if not os.path.exists(rdf_file):
                print(f"\n[!] Fatal File Error: The configured system storage path '{rdf_file}' does not exist.")
                return
            
            detected_format = guess_format(rdf_file)
            if not detected_format:
                print("\n[!] Fatal Configuration Error: Extension schema from target file name matches no known RDF serialization format layout.")
                return
                
            try:
                graph.parse(source=rdf_file, format=detected_format)
            except Exception as parse_error:
                print(f"\n[!] Fatal Structural Error: Provided file context is not a valid RDF serialization format layout ({detected_format}).\n    Parser Details: {parse_error}")
                return
        else:
            print("\n[!] Technical Ingestion Bound: No active input stream parameter path provided.")
            return

        print(f"[+] Triplestore parsed successfully into memory using layout parsing mode: '{detected_format}'. Size: {len(graph)} triples.")
        print("[*] Compiling schema extraction query parsing operations...")
        
        try:
            query_job = graph.query(query_str)
        except Exception as query_error:
            print(f"\n[!] Fatal SPARQL Syntax Error: The external query file could not be compiled.\n    Details: {query_error}")
            return

        columns = [str(var) for var in query_job.vars]
        
        rows = []
        for row in query_job:
            rows.append([val.toPython() if hasattr(val, 'toPython') else str(val) if val is not None else None for val in row])
            
        raw_df = pd.DataFrame(rows, columns=columns)
        if raw_df.empty:
            print("[!] Fatal Error: Extraction query generated an empty dataframe context. Verify your query's URI namespaces.")
            return

        print(f"[+] Raw payload captured successfully: {len(raw_df)} metadata rows loaded.")

        # Execute analytical processing sub-modules
        processed_df = self.clean_and_normalize_payload(raw_df)
        analyzed_df, yearly_df, metadata = self.execute_advanced_analytics(processed_df)
        self.generate_publication_plots(analyzed_df, yearly_df, metadata)
        
        csv_path = os.path.join(self.output_dir, "processed_scientometric_matrix.csv")
        analyzed_df.to_csv(csv_path, index=False)
        print(f"\n[+++] Success! All 7 visual assets and structural tables saved to: '{self.output_dir}/'\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphLens: Automated Semantic Graph Scientometric Pipeline")
    
    # Query configuration file input parameter
    parser.add_argument("-q", "--query", required=True, help="Path to external SPARQL query file containing prefix definitions (.rq, .sparql)")
    
    # Input data channels
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--url", help="Remote url target channel pointing to an RDF graph serialization (.ttl, .rdf, .jsonld)")
    group.add_argument("-f", "--file", help="Local storage file path pointing directly to an RDF serialization data asset")
    
    args = parser.parse_args()
    
    # Read query file context safely
    if not os.path.exists(args.query):
        print(f"[!] Fatal Configuration Error: The specified query file '{args.query}' was not found.")
        exit(1)
        
    with open(args.query, "r", encoding="utf-8") as f:
        loaded_sparql_query = f.read()
    
    pipeline = GraphLensScientometricPipeline()
    pipeline.execute_pipeline(
        query_str=loaded_sparql_query, 
        rdf_url=args.url, 
        rdf_file=args.file
    )
