# Conversational Toxicity Prediction

A research proof-of-concept for analyzing toxicity in Twitter/X conversations using **NLP, temporal interaction graphs, transformer-based text generation, and toxicity classification**.

**Presented at ICMLT 2026**

> This repository is for academic experimentation. The results are preliminary and should not be treated as production-level moderation performance.

---

## Overview

The project builds a pipeline that:

1. Collects Twitter/X posts and replies using Selenium.
2. Cleans text and extracts sentiment, named entities, and BERT tokens.
3. Computes heuristic toxicity/hate-related features and engagement-based virality scores.
4. Detects unusual activity periods using rolling statistics over sentiment, toxicity, volume, and virality.
5. Adds keyword-based contextual categories and optional external news context from The Guardian API.
6. Builds a directed NetworkX graph representing reply and conversation-flow relationships over time.
7. Generates text using GPT-2 or DistilGPT-2.
8. Classifies generated and observed text using toxicity classifiers.
9. Compares the experimental configurations using classification and error metrics.

### Implementation note

The notebook is named `04_tgn_prediction.ipynb` and labels its main experiment **TGN + GPT-2 + ETHOS**.

However, the current implementation does **not** contain a learned Temporal Graph Network architecture. It constructs a timestamped `NetworkX` interaction graph and computes graph-derived features.

Similarly, the current GPT-2 generation function uses tweet text as its prompt. Learned graph embeddings are not directly passed into GPT-2.

Therefore, this repository is best understood as a **temporal graph-based experimental pipeline**, rather than an end-to-end learned TGN model.

---

## Pipeline

```text
Twitter/X Data Collection
          |
          v
NLP Preprocessing
(cleaning + sentiment + NER + BERT tokenization)
          |
          v
Toxicity / Spike Analysis
(heuristic hate score + engagement + virality)
          |
          v
Context Enrichment
(categories + optional Guardian news search)
          |
          v
Temporal Interaction Graph
(NetworkX: reply + conversation-flow edges)
          |
          v
Text Generation
(GPT-2 / DistilGPT-2)
          |
          v
Toxicity Classification
(RoBERTa toxicity classifier / DeHateBERT)
          |
          v
Experimental Evaluation
```

---

## Repository Structure

| File                                       | Purpose                                                                                                           |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `01_data_collection.py`                    | Selenium-based Twitter/X collection of top-level posts and replies                                                |
| `02_preprocessing_and_visualization.ipynb` | Text cleaning, sentiment analysis, NER, BERT tokenization, and visualizations                                     |
| `03_spike_analyser.py`                     | Heuristic toxicity scoring, virality scoring, spike detection, content categorization, and Guardian API context   |
| `04_tgn_prediction.ipynb`                  | Temporal interaction graph construction, graph analysis, text generation, toxicity classification, and evaluation |
| `data:/concatenated_dataset_final.csv`     | Processed project dataset currently stored in the repository                                                      |


---

## 1. Data Collection

### `01_data_collection.py`

The scraper uses **Selenium + Chrome WebDriver** and supports:

* Topic-based search
* Start/end date filtering
* Optional English-only filtering
* A limit on top-level posts
* A limit on replies collected per post
* Headless browser execution
* Tweet/reply metadata and engagement metrics

Collected information includes tweet IDs, usernames, conversation IDs, parent tweet IDs, timestamps, text, likes, retweets/shares, and reply counts.

Example:

```bash
python 01_data_collection.py \
    --topic "Black Lives Matter" \
    --start_date 2020-06-01 \
    --end_date 2020-06-30 \
    --output tweets.csv \
    --limit 200 \
    --replies_limit 50 \
    --headless \
    --english_only
```

**Note:** The scraper depends on Twitter/X page structure and Selenium selectors, so it is experimental and may require changes if the platform UI changes.

---

## 2. NLP Preprocessing

### `02_preprocessing_and_visualization.ipynb`

The notebook performs:

* Lowercasing
* URL removal
* Special-character removal
* Stopword removal
* Duplicate removal
* Empty-text filtering
* Sentiment scoring using TextBlob polarity
* Named entity extraction using spaCy `en_core_web_sm`
* BERT tokenization using `bert-base-uncased`
* Exploratory visualizations

The processed data is saved as CSV for subsequent analysis.

---

## 3. Toxicity and Spike Analysis

### `03_spike_analyser.py`

`ImprovedHybridSpikeAnalyzer` adds heuristic toxicity, engagement, contextual, and temporal features.

### Heuristic toxicity / hate score

The analyzer combines:

* Hate/toxicity-related keyword matches
* Toxicity amplifiers
* Emotional intensifiers
* Capitalization intensity
* Exclamation-mark intensity

The resulting `enhanced_hate_score` is a **heuristic signal**, not a human-annotated toxicity label.

### Virality score

The implementation calculates:

```text
likes + 3 × shares + 2 × comments
```

and normalizes it by dividing by 1000 and clipping the result to the range 0–1.

### Spike detection

Posts are aggregated by day and analyzed using rolling statistics over:

* Sentiment
* Enhanced hate score
* Post volume
* Virality

A day is marked as a spike when at least one of the implemented spike conditions is triggered.

### Context categories

Posts are assigned keyword-based categories including:

* Politics
* Immigration
* Protest
* Terrorism
* Crime
* Economy
* International
* Social
* General

### External context

The analyzer can use **The Guardian Open Platform API** to retrieve contextual news information around detected spike periods.

This provides contextual information; it does **not establish that a particular news event caused a spike**.

---

## 4. Temporal Interaction Graph

### `04_tgn_prediction.ipynb`

The notebook constructs a directed **NetworkX** graph from the conversation data.

### Nodes

Nodes represent tweets/replies and store information such as:

* Tweet text
* Timestamp
* Speaker
* Sentiment
* Toxicity/hate-related features
* Engagement
* Parent count
* Conversation centrality
* Amplifier/test indicators

### Edges

The implementation creates two relationship types:

* `reply` — connects a reply to its parent tweet
* `conversation_flow` — connects consecutive tweets within a conversation

Edges also store temporal and derived values such as hate correlation, hate propagation, and influence-related quantities.

The notebook also performs a time-based train/test split and analyzes graph properties such as conversation structure and centrality.

---

## 5. Text Generation and Toxicity Classification

The notebook loads:

* **GPT-2** (`gpt2`)
* **DistilGPT-2** (`distilgpt2`)
* **RoBERTa toxicity classifier** (`s-nlp/roberta_toxicity_classifier`)
* **DeHateBERT** (`Hate-speech-CNERG/dehatebert-mono-english`)

For each evaluated node, the current pipeline uses the tweet text as the generation prompt, generates a continuation, and then classifies generated and observed text.

The notebook evaluates three configurations:

1. `TGN+GPT2+ETHOS` — the notebook's main experimental configuration.
2. `DistilGPT2 + ETHOS` — a simpler generation baseline.
3. An alternative classifier using DeHateBERT.

The first configuration should be understood as the project's **graph-based experimental pipeline label**, not as a claim that a learned TGN neural architecture is being trained.

---

## Experimental Results

The following results are from the current notebook run on **94 test nodes**.

| Model configuration               |   MAE |  RMSE | Accuracy |    F1 |
| --------------------------------- | ----: | ----: | -------: | ----: |
| TGN+GPT2+ETHOS                    | 0.234 | 0.484 |    76.6% | 0.750 |
| DistilGPT2 + ETHOS                | 0.234 | 0.484 |    76.6% | 0.683 |
| DeHateBERT alternative classifier | 0.128 | 0.357 |    87.2% | 0.847 |

These results are experimental and should not be interpreted as a definitive benchmark. The evaluation set is relatively small.

The notebook also performs statistical comparisons between the experimental configurations.

---

## Dataset

The repository contains a processed dataset with approximately **3,800 Twitter/X posts and replies**.

The dataset contains fields covering:

* Tweet and conversation IDs
* Parent/reply relationships
* Timestamps and text
* Likes, shares, and comments
* Cleaned text
* Sentiment
* Named entities
* BERT token IDs
* Enhanced hate score
* Hate intensity
* Virality score
* Spike/context features
* User influence
* Conversation centrality
* Reply-chain information

The repository dataset and the smaller evaluation set used in the notebook should not be treated as the same dataset.

---

## Technologies

**Data Collection**

* Python
* Selenium
* Chrome WebDriver
* WebDriver Manager
* `langdetect`

**NLP**

* spaCy
* TextBlob
* NLTK
* BERT tokenizer
* Transformers
* PyTorch

**Text Generation**

* GPT-2
* DistilGPT-2

**Toxicity Classification**

* RoBERTa-based toxicity classifier
* DeHateBERT

**Graphs**

* NetworkX

**Data Analysis**

* Pandas
* NumPy
* SciPy
* scikit-learn
* Matplotlib
* Seaborn
* WordCloud

**External Context**

* The Guardian Open Platform API

---

## Installation

Install the main dependencies:

```bash
pip install pandas numpy scipy scikit-learn \
    selenium webdriver-manager langdetect tqdm \
    nltk textblob spacy transformers torch \
    networkx matplotlib seaborn wordcloud \
    beautifulsoup4 requests
```

Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

The NLTK-based spike analyzer downloads its required NLTK resources when they are missing.

---

## Running the Project

### 1. Collect data

Run `01_data_collection.py` with the topic/date arguments shown above.

### 2. Preprocess data

Open:

```text
02_preprocessing_and_visualization.ipynb
```

and run the notebook cells.

### 3. Run spike analysis

Use:

```text
03_spike_analyser.py
```

with the appropriate input dataset/API configuration.

### 4. Run graph and prediction experiments

Open:

```text
04_tgn_prediction.ipynb
```

and run the notebook cells.

> The notebooks contain project-specific input/output filenames from the original experiments, so paths may need to be adjusted when reproducing the pipeline from scratch.

---

## Limitations

* The toxicity/hate score is heuristic and is not a human-annotated moderation label.
* The dataset and evaluation set are relatively small.
* GPT-2 generated text may not represent an actual future conversational response.
* The graph implementation uses NetworkX and engineered graph features rather than a learned end-to-end TGN architecture.
* The current generation function uses tweet text as the prompt rather than injecting learned graph embeddings into GPT-2.
* The Twitter/X scraper depends on platform-specific Selenium selectors.
* External news retrieval provides context but does not prove causality.
* The project is not designed for real-time or production moderation.

---

## Future Work

Possible extensions include:

* Training a genuine temporal graph neural network/TGN architecture
* Feeding learned graph representations into the prediction or generation model
* Larger and more diverse datasets
* Human-annotated toxicity labels
* More rigorous temporal forecasting evaluation
* Multilingual toxicity analysis
* Explainable graph-based predictions
* Real-time/streaming graph processing
* Better external-event attribution

---

## Research Context

This repository accompanies the research work:

**"Conversational Toxicity Prediction: A TGN-Augmented Transformer Approach"**

Presented at **ICMLT 2026**.

The repository contains an updated experimental run, so its reported metrics may differ from those in the original paper.

---

## Disclaimer

This repository is intended for **academic research and experimentation only**.

The toxicity and hate-intensity scores produced by the system are heuristic or model-derived signals and should not be treated as definitive moderation labels.

The system is **not intended for automated enforcement or production moderation** without substantially larger datasets, stronger validation, human evaluation, and appropriate safeguards.
