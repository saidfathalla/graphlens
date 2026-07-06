[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Build Status](https://github.com/saidfathalla/GraphLens/actions/workflows/pytest.yml/badge.svg)](https://github.com/saidfathalla/GraphLens/actions)
[![FAIR Principles](https://img.shields.io/badge/FAIR-Supported-brightgreen.svg)](https://www.go-fair.org/fair-principles/)
[![RDF/Semantic Web](https://img.shields.io/badge/Semantic-RDF-orange.svg)](https://www.w3.org/RDF/)

# GraphLens: Automated Semantic Graph Scientometric Pipeline

GraphLens is an automated, single-command command-line framework engineered to bridge the operational gap between structured semantic web repositories (RDF/Scholarly Knowledge Graphs) and advanced statistical scientometric analysis. 

Rather than requiring researchers to manually construct, debug, and optimize complex, resource-intensive federated SPARQL queries, GraphLens wraps the entire data engineering, feature vectorization, unsupervised machine learning, and visualization lifecycle within a headless execution loop.

## Features

* **Autonomous Layout Heuristics Engine**: Utilizes an integrated text-based machine learning pipeline (`CountVectorizer` + `DecisionTreeClassifier`) to bootstrap layout heuristics and classify column structures.
* **Multi-Dimensional Graph Ingestion**: Streams remote or local open-world triplestores natively via `rdflib`, executing graph-flattening query configurations to bypass nested property paths.
* **Mathematical Imputation & Cross-Derivation**: Automatically resolves missing open-world metadata literals by cross-calculating dependent peer-review statistics (e.g., mathematically inferring missing `accepted`, `submitted`, or `acceptanceRate` data points) to maximize observation sample sizes.
* **Unsupervised Taxonomic Profiling**: Applies $K$-means clustering ($K=3$) over continuous selectivity boundaries to automatically segment venues into standardized behavioral tiers (*Elite / Highly Selective*, *Standard Peer-Reviewed*, and *High-Acceptance / Mega-Venue*).
* **Longitudinal Growth Trajectory Regression**: Embeds parallel Ordinary Least Squares (OLS) linear regression configurations to compute annual growth velocities and trend fit coefficients ($R^2$) across multi-decade timeline horizons.
* **Automated Production Asset Materialization**: Systematically exports an entire suite of 7 high-density, journal-ready visualization plots (300 DPI, serif-typed) and a unified data matrix without manual plotting configurations.

## Installation

### Prerequisites
* Python 3.8+
* `pip` package manager
* Virtual environment framework (`venv` or `conda`, recommended)

### Core Dependencies
The pipeline engine automatically manages or builds upon the following foundational semantic and analytical runtimes:
* **`rdflib`**: For core W3C RDF parsing, graph manipulation, and Turtle serialization.
* **`pandas` & `numpy`**: For optimized structural array indexing, matrix cleaning, and data framing operations.
* **`scikit-learn`**: For internal ML layout classification, $K$-means data clustering, and OLS regression modeling.
* **`matplotlib` & `seaborn`**: For rendering high-impact, publication-ready statistical visualizations in headless environments.

### Install dependencies:
We recommend using a virtual environment.

```bash
pip install requests numpy pandas matplotlib seaborn rdflib scikit-learn
```

## Usage
GraphLens is designed as an automated, single-command command-line tool. Run the main script file to execute the pipeline:

```bash
python3 graphlens.py
```

By default, the pipeline automatically connects to the live open-science E[VENTSKG-Dataset](https://raw.githubusercontent.com/saidfathalla/EVENTSKG-Dataset/refs/heads/master/EVENTSKG.ttl) turtle data channel, 
compiles the semantic schemas, runs the advanced analytics core, and routes all outputs directly to `output/`.


