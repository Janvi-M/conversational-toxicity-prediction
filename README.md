# Conversational Toxicity Prediction

A research project for analyzing and predicting toxicity in online conversations using **NLP, temporal interaction graphs, transformer-based text generation, and toxicity classification**.

**Presented at ICMLT 2026**

## Overview

The project analyzes conversational activity on Twitter/X to identify toxicity spikes, understand their temporal and interaction patterns, and evaluate transformer-based approaches for conversational toxicity prediction.

### Pipeline

```text
Twitter/X Data Collection
        ↓
NLP Preprocessing & Sentiment Analysis
        ↓
Toxicity / Virality / Spike Detection
        ↓
External Context Analysis
        ↓
Temporal Interaction Graph
        ↓
GPT-2 / DistilGPT-2
        ↓
Toxicity Classification
        ↓
Evaluation & Baseline Comparison
```

## Key Features

* **Twitter/X conversational data collection** with support for topics, date ranges, replies, engagement metrics, and English-language filtering.
* **NLP preprocessing** using NLTK, TextBlob, spaCy, and BERT tokenization.
* **Toxicity and virality analysis** using heuristic hate-intensity scoring and engagement-based virality scoring.
* **Temporal spike detection** using sentiment, toxicity, conversation volume, and virality trends.
* **Contextual analysis** using entity extraction, content categorization, and Guardian News API correlation.
* **Temporal interaction graph construction** using NetworkX with timestamped nodes and reply/conversation-flow edges.
* **Graph-based features** including hate intensity, user influence, engagement, conversation depth, centrality, and spike metadata.
* **Transformer-based generation** using GPT-2 and DistilGPT-2.
* **Toxicity classification** using ETHOS, RoBERTa-based toxicity classification, and DeHateBERT.
* **Baseline comparison** against Ridge Regression, SVR, Random Forest, and Gradient Boosting.

## Experimental Results

Evaluation was performed on a held-out temporal test set of **94 conversational nodes**.

| Configuration                  |       MAE |      RMSE |  Accuracy |        F1 |
| ------------------------------ | --------: | --------: | --------: | --------: |
| TGN + GPT-2 + ETHOS            |     0.234 |     0.484 | **76.6%** | **0.750** |
| DistilGPT-2 + ETHOS            |     0.234 |     0.484 |     76.6% |     0.683 |
| GPT-2 + Alternative Classifier | **0.128** | **0.357** | **87.2%** | **0.847** |

> **Note:** "TGN + GPT-2 + ETHOS" refers to the project's temporal graph-based experimental pipeline. The repository constructs a temporal interaction graph with engineered graph features rather than implementing a learned neural TGN architecture.

## Repository Structure

```text
├── 01_data_collection.py
├── 02_preprocessing_and_visualization.ipynb
├── 03_spike_analyser.py
├── 04_tgn_prediction.ipynb
└── README.md
```

Raw Twitter/X data is **not included** in this repository.

## Technologies

**Python · PyTorch · NetworkX · Transformers · GPT-2 · DistilGPT-2 · BERT · RoBERTa · DeHateBERT · NLTK · spaCy · TextBlob · Scikit-learn · Pandas · NumPy**

## Limitations

* Toxicity and spike scores include heuristic components.
* Evaluation is performed on a relatively small test set.
* The temporal graph is implemented using NetworkX and engineered graph features rather than a learned TGN architecture.
* GPT-2 generation is conditioned on conversational text rather than learned graph embeddings.
* External news correlations indicate contextual relevance, not causation.
* The project is intended as a research proof-of-concept, not a production moderation system.

## Research

**Presented at ICMLT 2026**
Research focus: *Conversational Toxicity Prediction using Temporal Interaction Analysis and Transformer-based Models.*
