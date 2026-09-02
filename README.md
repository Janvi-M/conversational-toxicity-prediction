# Conversational Toxicity Prediction

A research proof-of-concept for analyzing toxicity in online conversations using **NLP, temporal interaction graphs, transformer-based text generation, and toxicity classification**.

**Presented at ICMLT 2026**

> **Implementation note:** The current repository constructs a timestamped interaction graph using NetworkX and derives graph-based features for analysis and prediction. It does **not** implement a learned Temporal Graph Network (TGN) architecture.

---

## Overview

Online conversations can exhibit sudden increases in toxic or hateful content, often influenced by rapidly developing events and interactions between users.

This project explores a pipeline for:

* Collecting Twitter/X conversational data
* Preprocessing and analyzing text
* Detecting toxicity and conversational spikes using heuristic methods
* Enriching spike periods with external news context
* Constructing temporal interaction graphs
* Generating conversational text using GPT-2 and DistilGPT-2
* Classifying generated/content text for toxicity
* Evaluating different model configurations

The project was developed as a research proof-of-concept rather than a production moderation system.

---

## Pipeline

```text
Twitter/X Data Collection
          ↓
NLP Preprocessing
          ↓
Toxicity & Spike Analysis
          ↓
Context Enrichment
          ↓
Temporal Interaction Graph
          ↓
GPT-2 / DistilGPT-2
          ↓
Toxicity Classification
          ↓
Evaluation
```

---

## Repository Structure

```text
.
├── 01_data_collection.py
├── 02_preprocessing_and_visualization.ipynb
├── 03_spike_analyser.py
├── 04_tgn_prediction.ipynb
├── README.md
└── data/
```

The repository does **not** include the collected Twitter/X dataset.

---

## 1. Data Collection

`01_data_collection.py`

The data collection component was designed to collect conversational Twitter/X data for research analysis.

The collected data included tweet-level conversational information and metadata required for downstream temporal and interaction analysis.

Raw collected data is **not included in this repository**.

---

## 2. Preprocessing & Visualization

`02_preprocessing_and_visualization.ipynb`

The preprocessing pipeline includes:

* Lowercasing text
* URL removal
* Special-character cleaning
* Stopword removal
* TextBlob sentiment/polarity analysis
* spaCy-based linguistic processing
* BERT tokenization using `bert-base-uncased`

The notebook also performs exploratory visualization and analysis of the processed data.

---

## 3. Toxicity & Spike Analysis

`03_spike_analyser.py`

The project uses an enhanced heuristic approach to identify potentially toxic/hateful content and periods of increased activity.

The analyzer incorporates:

* Hate-related keywords
* Toxicity amplifiers
* Emotional intensifiers
* Capitalization intensity
* Exclamation intensity
* Contextual categories such as:

  * Politics
  * Immigration
  * Protest
  * Terrorism
  * Crime
  * Economy
  * International events
  * Social issues
* Entity extraction using NLP and regular expressions
* Virality scoring based on engagement signals
* Daily spike detection

For detected spikes, the pipeline can retrieve related news context through the **Guardian API**.

> The toxicity score is heuristic and should not be interpreted as a human-annotated toxicity label.

---

## 4. Temporal Interaction Graph

`04_tgn_prediction.ipynb`

The project represents conversational interactions as a **timestamped NetworkX graph**.

The graph captures relationships such as:

* Reply interactions
* Conversation-flow relationships

Graph analysis includes temporal train/test splitting and graph-derived features such as network connectivity and centrality measures.

### Important Implementation Note

Despite the original experimental naming of the notebook and model configuration, this repository does **not** contain a learned Temporal Graph Network architecture.

Instead, it uses:

* NetworkX for temporal interaction graph construction
* Engineered graph features
* Temporal splitting for evaluation
* Transformer-based language models for text generation
* Toxicity classifiers for classification

This distinction is important when interpreting the reported experimental results.

---

## 5. Transformer Models

The project experiments with transformer-based language models including:

### GPT-2

Used for conversational text generation.

### DistilGPT-2

A smaller GPT-2 variant used as an alternative generation model.

The current implementation uses tweet text as the generation prompt. Learned graph embeddings are **not directly passed into GPT-2**.

---

## 6. Toxicity Classification

The project evaluates multiple toxicity/hate-speech classification approaches, including:

* **RoBERTa toxicity classifier**

  * `s-nlp/roberta_toxicity_classifier`
* **DeHateBERT**

  * `Hate-speech-CNERG/dehatebert-mono-english`

These models are used to assess toxicity in the generated or analyzed text.

---

## 7. Experimental Results

The current notebook reports results on a held-out temporal test set of **94 test nodes**.

| Model Configuration               |   MAE |  RMSE | Accuracy |    F1 |
| --------------------------------- | ----: | ----: | -------: | ----: |
| TGN + GPT-2 + ETHOS               | 0.234 | 0.484 |    76.6% | 0.750 |
| DistilGPT-2 + ETHOS               | 0.234 | 0.484 |    76.6% | 0.683 |
| DeHateBERT Alternative Classifier | 0.128 | 0.357 |    87.2% | 0.847 |

> **Note:** `TGN + GPT-2 + ETHOS` is the original experimental configuration label. The repository implementation uses a temporal NetworkX graph and engineered graph features rather than a learned TGN architecture.

---

## Limitations

This project has several limitations:

* The toxicity/spike detection component is primarily heuristic.
* The evaluation dataset is relatively small.
* The current implementation does not use a learned TGN architecture.
* Graph representations are based on NetworkX and engineered features.
* Graph embeddings are not directly incorporated into GPT-2 generation.
* GPT-2 generation is prompted using tweet text and is not guaranteed to represent a true future conversational response.
* Twitter/X data collection can be brittle due to changes in platform access and interfaces.
* External news context provides correlation/context rather than proof of causality.
* The system is not intended for production-scale content moderation.

---

## Future Work

Potential extensions include:

* Implementing a true learned Temporal Graph Network
* Learning graph representations end-to-end
* Integrating learned graph embeddings with language models
* Increasing the size and diversity of the evaluation data
* Using human-annotated toxicity labels
* More rigorous temporal forecasting experiments
* Multilingual toxicity detection
* Explainable toxicity prediction
* Real-time/streaming conversational analysis
* Better attribution of toxicity spikes to external events

---

## Technologies

**Programming & ML**

* Python
* PyTorch
* Scikit-learn
* Pandas
* NumPy

**NLP & Transformers**

* GPT-2
* DistilGPT-2
* RoBERTa
* DeHateBERT
* BERT Tokenizer
* spaCy
* TextBlob

**Graph Analysis**

* NetworkX
* Temporal interaction graphs
* Graph-based features

**Data & APIs**

* Twitter/X data collection
* Guardian API

---

## Research

This work was presented at **ICMLT 2026** as part of research on conversational toxicity prediction and temporal interaction analysis.

---

## Disclaimer

This repository is intended for **research and educational purposes**. The implemented toxicity detection methods should not be treated as a definitive measure of harmful content or as a production-ready moderation system.
