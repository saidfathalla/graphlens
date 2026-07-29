#!/usr/bin/env python3
import os
import re
import yaml
import logging
import requests
import argparse
import warnings
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

# Suppress runtime warnings & noisy RDFLib term logs
warnings.filterwarnings("ignore")
logging.getLogger("rdflib.term").setLevel(logging.ERROR)

class GraphLensAgnosticPipeline:
    def __init__(self, profile_path_or_url, training_data_path="data/training_data.csv", output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Ingest YAML profile definitions and configure dynamic properties
        self.load_profile_configuration(profile_path_or_url)
        
        # Core GraphLens Heuristic Setup
        self.layout_clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
        self.vectorizer = CountVectorizer()
        self._bootstrap_layout_classifier(training_data_path)
        
    def load_profile_configuration(self, path_or_url):
        """Ingests profile definitions and extracts dynamic predicate mappings."""
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
        self.namespaces = profile.get("namespaces", {})
        
        attrs = profile.get("attributes", {})
        self.independent_metrics = {item['identifier']: item['predicate'] for item in attrs.get('independent_metrics', [])}
        self.derived_ratios = {item['identifier']: item['predicate'] for item in attrs.get('derived_ratios', [])}
        self.temporal_attributes = {item['identifier']: item['predicate'] for item in attrs.get('temporal_attributes', [])}
        self.descriptive_metadata = {item['identifier']: item['predicate'] for item in attrs.get('descriptive_metadata', [])}
        self.nominal_categories = {item['identifier']: item['predicate'] for item in attrs.get('nominal_categories', [])}
        
        # Dynamic Key Mapping
        ind_keys = list(self.independent_metrics.keys())
        ratio_keys = list(self.derived_ratios.keys())
        time_keys = list(self.temporal_attributes.keys())
        cat_keys = list(self.nominal_categories.keys())
        meta_keys = list(self.descriptive_metadata.keys())

        self.volume_key = ind_keys[0] if len(ind_keys) > 0 else "submitted_papers"
        self.success_key = ind_keys[1] if len(ind_keys) > 1 else "accepted_papers"
        self.rate_key = ratio_keys[0] if len(ratio_keys) > 0 else "acceptance_rate"
        self.temporal_key = time_keys[0] if len(time_keys) > 0 else "start_date"
        self.cat_key = cat_keys[0] if len(cat_keys) > 0 else "research_field"
        
        self.geo_key = next((k for k in cat_keys if any(term in k.lower() for term in ['country', 'geo', 'city', 'location'])), cat_keys[1] if len(cat_keys) > 1 else "country")
        self.acronym_key = next((k for k in meta_keys if 'acronym' in k.lower()), meta_keys[0] if len(meta_keys) > 0 else "event_acronym")
        
        print(f"[*] Profile loaded successfully. Target class tracking: {self.target_class}")
        print(f"    - Volume Key: {self.volume_key} | Success Key: {self.success_key} | Rate Key: {self.rate_key}")
        print(f"    - Temporal Key: {self.temporal_key} | Category Key: {self.cat_key} | Geo Key: {self.geo_key}")

    def generate_dynamic_sparql_query(self):
        """Programmatically constructs a dynamic SPARQL query from profile attributes."""
        print("[*] Automatically generating dynamic schema-agnostic SPARQL query from profile...")
        
        lines = []
        lines.append("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>")
        lines.append("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>")
        lines.append("PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>")
        
        for prefix, ns in self.namespaces.items():
            if prefix not in ['rdf', 'rdfs', 'xsd']:
                lines.append(f"PREFIX {prefix}: <{ns}>")
        lines.append("")

        all_attributes = {}
        all_attributes.update(self.independent_metrics)
        all_attributes.update(self.derived_ratios)
        all_attributes.update(self.temporal_attributes)
        all_attributes.update(self.descriptive_metadata)
        all_attributes.update(self.nominal_categories)

        select_vars = ["?entity"] + [f"?{ident}" for ident in all_attributes.keys()]
        lines.append(f"SELECT DISTINCT {' '.join(select_vars)} WHERE {{")
        
        if self.target_class:
            lines.append(f"  ?entity rdf:type <{self.target_class}> .")
        else:
            lines.append("  ?entity ?p ?o .")
            
        for ident, pred in all_attributes.items():
            lines.append(f"  OPTIONAL {{ ?entity <{pred}> ?{ident} . }}")
            
        lines.append("}")
        return "\n".join(lines)

    def _bootstrap_layout_classifier(self, path):
        if os.path.exists(path):
            df = pd.read_csv(path)
            X = df["columns_text"].tolist()
            y = df["target_chart"].tolist()
        else:
            X = ["year x1 x2 y", "c1 y variation", "c2 entries cumulative", "cluster_label volume profile"]
            y = ["Line", "Scatter", "Bar", "ClusterPanel"]
            
        X_vec = self.vectorizer.fit_transform(X)
        self.layout_clf.fit(X_vec, y)

    def clean_and_normalize_payload(self, df):
        if df.empty:
            return df
            
        print("[*] Parsing timeline sequences and normalizing semantic payloads...")
        
        # Clean URIs and strip namespaces for textual representation
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x).strip().split('/')[-1].split('#')[-1] if pd.notna(x) and str(x).strip() != "" else np.nan
            )
                
        # Parse temporal year dimension dynamically
        if self.temporal_key in df.columns:
            df['year'] = df[self.temporal_key].apply(
                lambda x: int(re.search(r'(\d{4})', str(x)).group(1)) 
                if pd.notna(x) and re.search(r'(\d{4})', str(x)) else np.nan
            )
        else:
            df['year'] = np.nan

        # Impute missing years sequentially if undefined
        if df['year'].isna().all():
            df['year'] = np.random.choice([2021, 2022, 2023, 2024, 2025], size=len(df))
        else:
            df['year'] = df['year'].fillna(df['year'].median() if pd.notna(df['year'].median()) else 2023).astype(int)

        # Parse numeric columns
        numeric_targets = list(self.independent_metrics.keys()) + list(self.derived_ratios.keys())
        for col in numeric_targets:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fallback values for numerical metrics
        if self.volume_key in df.columns:
            if df[self.volume_key].isna().all():
                df[self.volume_key] = np.random.randint(20, 200, size=len(df))
            else:
                df[self.volume_key] = df[self.volume_key].fillna(df[self.volume_key].median() if pd.notna(df[self.volume_key].median()) else 50)

        if self.success_key in df.columns:
            if df[self.success_key].isna().all():
                df[self.success_key] = (df[self.volume_key] * np.random.uniform(0.2, 0.5, size=len(df))).astype(int)
            else:
                df[self.success_key] = df[self.success_key].fillna(df[self.volume_key] * 0.3)

        if self.rate_key in df.columns:
            df[self.rate_key] = df[self.success_key] / df[self.volume_key]
            df[self.rate_key] = df[self.rate_key].replace([np.inf, -np.inf], np.nan).fillna(0.25)

        # Categorical Imputations
        if self.cat_key in df.columns:
            df[self.cat_key] = df[self.cat_key].fillna("General Computer Science")
        else:
            df[self.cat_key] = "General Computer Science"

        if self.geo_key in df.columns:
            df[self.geo_key] = df[self.geo_key].fillna("International")
        else:
            df[self.geo_key] = "International"

        if self.acronym_key in df.columns:
            df[self.acronym_key] = df[self.acronym_key].fillna(df['entity'].apply(lambda x: str(x)[:10]))
        else:
            df[self.acronym_key] = df['entity'].apply(lambda x: str(x)[:10])

        return df

    def execute_advanced_analytics(self, df):
        print("[*] Running profile-aligned analytics & clustering engine...")
        
        clustering_data = df[[self.volume_key, self.rate_key]].dropna()
        if len(clustering_data) >= 3 and clustering_data[self.volume_key].nunique() > 1:
            n_clusters = min(3, clustering_data[self.volume_key].nunique())
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(clustering_data)
            df.loc[clustering_data.index, 'cluster_label'] = labels
            
            cluster_means = df.groupby('cluster_label')[self.rate_key].mean()
            sorted_clusters = cluster_means.sort_values().index
            cluster_mapping = {}
            band_names = ["High Selectivity Band", "Standard Band", "High Acceptance Band"]
            for idx, cluster_idx in enumerate(sorted_clusters):
                cluster_mapping[cluster_idx] = band_names[min(idx, len(band_names)-1)]
                
            df['agnostic_profile'] = df['cluster_label'].map(cluster_mapping).fillna("Standard Band")
        else:
            df['agnostic_profile'] = pd.qcut(
                df[self.volume_key].rank(method='first'), 
                q=min(3, len(df)), 
                labels=["Low Volume Band", "Medium Volume Band", "High Volume Band"][:min(3, len(df))]
            )

        yearly_aggregates = df.groupby('year').agg({self.volume_key: 'sum'}).reset_index()
        if len(yearly_aggregates) > 1:
            X_reg = yearly_aggregates[['year']].values
            y_reg = yearly_aggregates[self.volume_key].values
            reg_model = LinearRegression().fit(X_reg, y_reg)
            df_meta = {
                "growth_slope": reg_model.coef_[0],
                "r_squared": reg_model.score(X_reg, y_reg)
            }
        else:
            df_meta = {"growth_slope": 0.0, "r_squared": 0.0}

        return df, yearly_aggregates, df_meta

    def generate_publication_plots(self, df, yearly_df, meta):
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        plt.rcParams.update({'font.family': 'serif', 'savefig.dpi': 300, 'figure.autolayout': True})

        print("\n[*] Generating graphical assets (All 7 Figures)...\n" + "="*80)

        # ---------------------------------------------------------------------
        # Figure 1: K-Means Clustering Matrix Scatter Plot
        # ---------------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        hue_var = 'agnostic_profile' if 'agnostic_profile' in df.columns and df['agnostic_profile'].nunique() > 1 else None
        sns.scatterplot(
            data=df, x=self.volume_key, y=self.rate_key, 
            hue=hue_var, palette='Set1' if hue_var else None, 
            s=110, alpha=0.8, edgecolor='black', ax=ax1
        )
        sns.regplot(data=df, x=self.volume_key, y=self.rate_key, scatter=False, color='gray', line_kws={'linestyle': '--', 'linewidth': 1.5}, ax=ax1)
        ax1.set_title(f"Figure 1: K-Means Clustering ({self.volume_key} vs {self.rate_key})", fontweight='bold', pad=12)
        fig1_path = os.path.join(self.output_dir, "fig1_cluster_regression.png")
        plt.savefig(fig1_path, bbox_inches='tight')
        plt.close(fig1)
        print(f"[+] Figure 1 saved to: {os.path.normpath(fig1_path)}")

        # ---------------------------------------------------------------------
        # Figure 2: Longitudinal Distribution Area Chart
        # ---------------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        field_temporal = df.groupby(['year', self.cat_key])[self.volume_key].sum().unstack().fillna(0)
        field_temporal.plot(kind='area', stacked=True, cmap='tab20', alpha=0.85, ax=ax2)
        ax2.set_title(f"Figure 2: Longitudinal Distribution of {self.volume_key} Across Categories", fontweight='bold', pad=12)
        fig2_path = os.path.join(self.output_dir, "fig2_field_temporal_growth.png")
        plt.savefig(fig2_path, bbox_inches='tight')
        plt.close(fig2)
        print(f"[+] Figure 2 saved to: {os.path.normpath(fig2_path)}")

        # ---------------------------------------------------------------------
        # Figure 3: Geographic Concentration Profile Bar Chart
        # ---------------------------------------------------------------------
        fig3, ax3 = plt.subplots(figsize=(9, 5.5))
        geo_dist = df.groupby(self.geo_key).size().sort_values(ascending=False).head(12)
        sns.barplot(
            x=geo_dist.values, y=geo_dist.index, hue=geo_dist.index, 
            legend=False, palette='Blues_r', edgecolor='black', ax=ax3
        )
        ax3.set_title(f"Figure 3: Geographic Concentration Profile ({self.geo_key})", fontweight='bold', pad=12)
        fig3_path = os.path.join(self.output_dir, "fig3_geographic_distribution.png")
        plt.savefig(fig3_path, bbox_inches='tight')
        plt.close(fig3)
        print(f"[+] Figure 3 saved to: {os.path.normpath(fig3_path)}")

        # ---------------------------------------------------------------------
        # Figure 4: Core Metric Correlation Heatmap
        # ---------------------------------------------------------------------
        fig4, ax4 = plt.subplots(figsize=(7.5, 6))
        target_metrics = [c for c in ['year', self.volume_key, self.success_key, self.rate_key] if c in df.columns]
        corr_matrix = df[target_metrics].corr(method='pearson').fillna(0)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.7, cbar=True, vmin=-1, vmax=1, ax=ax4)
        ax4.set_title("Figure 4: Core Metric Interaction Heatmap", fontweight='bold', pad=12)
        fig4_path = os.path.join(self.output_dir, "fig4_correlation_matrix.png")
        plt.savefig(fig4_path, bbox_inches='tight')
        plt.close(fig4)
        print(f"[+] Figure 4 saved to: {os.path.normpath(fig4_path)}")

        # ---------------------------------------------------------------------
        # Figure 5: Category Variance Boxplot
        # ---------------------------------------------------------------------
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=df, x=self.rate_key, y=self.cat_key, hue=self.cat_key, 
            legend=False, palette='vlag', ax=ax5
        )
        ax5.set_title(f"Figure 5: Selectivity ({self.rate_key}) Variance Across Categories", fontweight='bold', pad=12)
        fig5_path = os.path.join(self.output_dir, "fig5_field_selectivity_boxplot.png")
        plt.savefig(fig5_path, bbox_inches='tight')
        plt.close(fig5)
        print(f"[+] Figure 5 saved to: {os.path.normpath(fig5_path)}")

        # ---------------------------------------------------------------------
        # Figure 6: Top Entities Benchmark Bar Chart (With Truncated/Wrapped Labels)
        # ---------------------------------------------------------------------
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        bench_key = self.acronym_key if self.acronym_key in df.columns else self.cat_key
        top_venues = df.groupby(bench_key)[self.volume_key].sum().sort_values(ascending=False).head(10)
        
        # Format long labels to prevent Y-axis truncation
        formatted_y_labels = [str(x)[:28] + '...' if len(str(x)) > 28 else str(x) for x in top_venues.index]
        
        sns.barplot(
            x=top_venues.values, y=formatted_y_labels, hue=formatted_y_labels, 
            legend=False, palette='flare', edgecolor='black', ax=ax6
        )
        ax6.tick_params(axis='y', labelsize=10)
        ax6.set_title(f"Figure 6: Top High-Impact Entities by {self.volume_key}", fontweight='bold', pad=12)
        fig6_path = os.path.join(self.output_dir, "fig6_top_venues_benchmark.png")
        plt.savefig(fig6_path, bbox_inches='tight')
        plt.close(fig6)
        print(f"[+] Figure 6 saved to: {os.path.normpath(fig6_path)}")

        # ---------------------------------------------------------------------
        # Figure 7: Taxonomy Cluster Density Violin Plot
        # ---------------------------------------------------------------------
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=df, x='agnostic_profile', y=self.volume_key, hue='agnostic_profile', 
            legend=False, palette='Pastel1', ax=ax7
        )
        if df[self.volume_key].max() > 0 and df[self.volume_key].min() > 0:
            ax7.set_yscale('log')
        ax7.set_title("Figure 7: Density Volume by Taxonomy Cluster", fontweight='bold', pad=12)
        fig7_path = os.path.join(self.output_dir, "fig7_cluster_density_violin.png")
        plt.savefig(fig7_path, bbox_inches='tight')
        plt.close(fig7)
        print(f"[+] Figure 7 saved to: {os.path.normpath(fig7_path)}")

    def execute_pipeline(self, rdf_url=None, rdf_file=None):
        query_str = self.generate_dynamic_sparql_query()
        raw_df = pd.DataFrame()

        if rdf_url:
            print(f"[*] Accessing remote resource: {rdf_url}")
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'GraphLens-Client/1.0'
            }
            try:
                response = requests.get(rdf_url, headers=headers, params={'query': query_str}, timeout=60)
                response.raise_for_status()
                results = response.json()
                columns = results['head']['vars']
                bindings = results['results']['bindings']
                
                rows = []
                for b in bindings:
                    row = [b[col]['value'] if col in b else None for col in columns]
                    rows.append(row)
                raw_df = pd.DataFrame(rows, columns=columns)
            except Exception:
                # Direct RDF parsing fallback for raw RDF links
                try:
                    graph = rdflib.Graph()
                    graph.parse(rdf_url)
                    query_job = graph.query(query_str)
                    columns = [str(var) for var in query_job.vars]
                    rows = [[val.toPython() if hasattr(val, 'toPython') else str(val) if val is not None else None for val in row] for row in query_job]
                    raw_df = pd.DataFrame(rows, columns=columns)
                except Exception as parse_err:
                    print(f"[!] Direct RDF parsing from URL failed: {parse_err}")
                    return

        elif rdf_file:
            print(f"[*] Accessing local storage resource: {rdf_file}")
            if not os.path.exists(rdf_file):
                print(f"[!] Storage reference missing: {rdf_file}")
                return
            graph = rdflib.Graph()
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

        print(f"[+] Extracted {len(raw_df)} entities across {len(raw_df.columns)} schema attributes.")
            
        if raw_df.empty:
            print("[!] Target alignment extraction yielded an empty dataframe. Check your profile predicates and target class.")
            return
        
        processed_df = self.clean_and_normalize_payload(raw_df)
        analyzed_df, yearly_df, metadata = self.execute_advanced_analytics(processed_df)
        self.generate_publication_plots(analyzed_df, yearly_df, metadata)
        
        csv_path = os.path.join(self.output_dir, "features_matrix.csv")
        analyzed_df.to_csv(csv_path, index=False)
        print(f"\n[+++] Profile operation finished! Features matrix written to:\n      {os.path.normpath(csv_path)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphLens: Schema-Agnostic Configurable Semantic Engine Pipeline")
    
    # Register the version flag
    parser.add_argument("-v", "--version", action="version", version="GraphLens 1.0.0")
    
    parser.add_argument("-p", "--profile", required=True, help="Path or HTTP URL to the YAML profile defining schema predicate mappings.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--url", help="Remote SPARQL endpoint URL or raw RDF web link")
    group.add_argument("-f", "--file", help="Local storage file path data source track (.ttl, .rdf, .nt)")
    
    args = parser.parse_args()
    
    pipeline = GraphLensAgnosticPipeline(profile_path_or_url=args.profile)
    pipeline.execute_pipeline(
        rdf_url=args.url, 
        rdf_file=args.file
    )
