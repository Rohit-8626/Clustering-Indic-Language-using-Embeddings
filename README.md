# Indic Language Clustering via BERT Embeddings

Unsupervised clustering of 8 Indian languages using IndicBERT embeddings, PCA-based dimensionality reduction, and multiple clustering algorithms. The project compares two generations of the IndicBERT model family (v1 vs v3) to evaluate which produces more separable language representations.

---

## Project Structure

```
├── data/
│   ├── raw/                  # Per-language CSVs downloaded from HuggingFace
│   ├── preprocess/           # Combined dataset (indic-language.csv)
│   ├── embeddings/           # Generated .npy embedding arrays
│   └── reduced/              # PCA-reduced .npy arrays
│
├── data_collection.ipynb              # Download & merge dataset
├── generate_embeddings_v1.ipynb       # Generate embeddings with IndicBERT v1
├── generate_embeddings_v3.ipynb       # Generate embeddings with IndicBERT-v3-4B
├── dimensionality_reduction_v1.ipynb  # PCA on v1 embeddings
├── dimensionality_reduction_v3.ipynb  # PCA on v3 embeddings
├── cluster_experiments_v1.ipynb       # Clustering experiments on v1
└── cluster_experiments_v3.ipynb       # Clustering experiments on v3
```

---

## Dataset

**Source:** [`ai4bharat/IndicHeadlineGeneration`](https://huggingface.co/datasets/ai4bharat/IndicHeadlineGeneration) on HuggingFace

**Languages (500 samples each):**

| Language   | Code |
|------------|------|
| Hindi      | hi   |
| Gujarati   | gu   |
| Punjabi    | pa   |
| Marathi    | mr   |
| Tamil      | ta   |
| Telugu     | te   |
| Malayalam  | ml   |
| Kannada    | kn   |

**Total samples:** 4,000 news article inputs across 8 classes

---

## Pipeline

```
Data Collection → Embedding Generation → PCA Reduction → Clustering → Evaluation
```

### 1. Data Collection (`data_collection.ipynb`)
Downloads 500 training samples per language from HuggingFace and merges them into a single labeled CSV.

### 2. Embedding Generation
Two models are compared:

| Notebook | Model | Embedding Dim |
|----------|-------|---------------|
| `generate_embeddings_v1.ipynb` | `ai4bharat/indic-bert` | 768 |
| `generate_embeddings_v3.ipynb` | `ai4bharat/IndicBERT-v3-4B` | 4096 |

Embeddings are extracted via mean pooling over the last hidden state (`float16`, batch size 32, max length 128 tokens, GPU accelerated).

### 3. Dimensionality Reduction
PCA is applied to both embedding sets, retaining components that explain **95% of variance**. Optimal component counts are plotted and saved as `.npy` arrays.

### 4. Clustering Experiments
Four algorithms are evaluated on both reduced embedding sets:

- **KMeans** — Silhouette score + elbow/inertia plots, k swept from 2–15
- **DBSCAN** — Epsilon selected via k-distance graph, MinPts = embedding dimension
- **Agglomerative Clustering** — Ward linkage with dendrogram visualization
- **Gaussian Mixture Model** — AIC/BIC curves to determine optimal components

Visualizations include 2D and interactive 3D t-SNE plots (via Plotly) and silhouette coefficient diagrams per cluster.

---

## Key Finding

**IndicBERT v1 produces significantly cleaner clusters than IndicBERT-v3-4B.**

Despite v3 being a much larger model, v1's embeddings yield higher silhouette scores and more visually separable language clusters. This suggests that for unsupervised clustering tasks using mean-pooled representations, model scale does not guarantee better geometric separation of language embeddings.

### Clustering Results Summary

| Algorithm | v1 Silhouette Score | v3 Silhouette Score |
|-----------|--------------------|--------------------|
| KMeans (best k) | ~0.28–0.32 | ~0.05–0.08 |
| DBSCAN | Separable clusters | Near-single cluster |
| Agglomerative | Clear 2-group split | Weak separation |
| GMM | AIC/BIC plateau at ~8 | No clear plateau |

> Note: Exact scores may vary due to t-SNE and KMeans stochasticity. Results above reflect the patterns observed across runs.

---

## Setup & Requirements

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (recommended for embedding generation)
- HuggingFace account with access token

### Installation

```bash
git clone https://github.com/your-username/indic-language-clustering.git
cd indic-language-clustering
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)

```
torch
transformers
datasets
huggingface_hub
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
scipy
tqdm
```

### HuggingFace Token

The dataset and models require a HuggingFace token. In each notebook, replace:

```python
login(token=your_token)
```

with your token, or set it as an environment variable:

```bash
export HF_TOKEN=your_token_here
```

### Running Order

Run the notebooks in this sequence:

```
1. data_collection.ipynb
2. generate_embeddings_v1.ipynb  (or v3)
3. dimensionality_reduction_v1.ipynb  (or v3)
4. cluster_experiments_v1.ipynb  (or v3)
```

> Update the data paths inside each notebook to match your local directory structure before running.

---

## Tools & Libraries

| Purpose | Library |
|---------|---------|
| Embeddings | `transformers`, `torch` |
| Data | `datasets`, `pandas`, `numpy` |
| Dimensionality Reduction | `scikit-learn` (PCA) |
| Clustering | `scikit-learn` (KMeans, DBSCAN, Agglomerative, GMM) |
| Visualization | `matplotlib`, `seaborn`, `plotly`, `scipy` |

---

## Limitations

- 500 samples per language is sufficient for a proof-of-concept but not for generalizable claims
- Mean pooling over `last_hidden_state` does not account for padding tokens in the average
- No baseline comparison (e.g., TF-IDF + clustering) was performed
- The reason v1 outperforms v3 on clustering is not yet analyzed (dimensionality vs. training objective vs. pooling strategy)

---

## Author

**Rohit Vastani**  
[GitHub](https://github.com/your-username) · [LinkedIn]([https://linkedin.com/in/rohit-vastani](https://www.linkedin.com/in/rohit-vastani-3a9a18301?utm_source=share_via&utm_content=profile&utm_medium=member_android))
