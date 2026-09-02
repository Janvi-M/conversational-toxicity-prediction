# Conversational Toxicity Prediction: A TGN-Augmented Transformer Approach

A research proof-of-concept for analyzing and predicting the evolution of toxicity in Twitter/X conversations using **NLP, temporal graph modeling, transformer-based language generation, and toxicity classification**.

**Presented at ICMLT 2026**

---

## Overview

Online conversations can become increasingly toxic as they evolve, particularly around socially and politically sensitive events. Traditional toxicity classifiers generally evaluate individual posts in isolation and may miss the **temporal and conversational context** surrounding them.

This project explores a context-aware approach that combines:

- Twitter/X conversation data and reply threads
- NLP preprocessing and feature extraction
- Temporal toxicity/spike analysis
- Conversational graph construction using Temporal Graph Network (TGN) concepts
- GPT-2-based future tweet generation
- ETHOS-based toxicity classification
- External news context using The Guardian Open Platform API

The goal is to investigate whether **conversation structure and temporal context can complement text-based toxicity analysis when estimating the toxicity of future conversational turns**.

> **This repository is a research proof of concept. The reported results are preliminary and should not be interpreted as production-level moderation performance.**

---

## Pipeline

```text
Twitter/X Data Collection
          │
          ▼
NLP Preprocessing
(cleaning • sentiment • NER • BERT tokenization)
          │
          ▼
Toxicity / Spike Analysis
(hate-score • sentiment • engagement • virality)
          │
          ▼
External Context
(The Guardian API + topic categories)
          │
          ▼
Temporal Graph Construction
(reply / mention / retweet relationships)
          │
          ▼
Future Tweet Generation
(GPT-2 / DistilGPT-2)
          │
          ▼
Toxicity Classification
(ETHOS / RoBERTa)
          │
          ▼
Comparison with Actual Future Tweets
```

---

# Repository Structure

| File | Description |
|------|-------------|
| `01_data_collection.py` | Selenium-based Twitter/X scraper for topic and date-range searches, including reply collection |
| `02_preprocessing_and_visulaization.ipynb` | Text preprocessing, sentiment analysis, NER, BERT tokenization, and exploratory visualization |
| `03_spike_analysis.py` | Toxicity/spike analysis, contextual categorization, hate-score and virality feature engineering, and Guardian API integration |
| `04_tgn_prediction.ipynb` | Temporal graph construction, future tweet generation, toxicity classification, and model evaluation |
| `data/concatenated_dataset_final.csv` | Final concatenated dataset containing processed, spike-related, engagement, and graph-derived features |

---

# Methodology

## 1. Twitter/X Data Collection

### `01_data_collection.py`

The data collection pipeline uses **Selenium and Chrome WebDriver** to collect Twitter/X posts for a specified topic and date range.

The scraper supports:

- Topic-based search
- Date-range filtering
- English-language filtering
- Configurable number of top-level posts
- Configurable number of replies per post
- Headless browser execution
- Reply-thread collection

Example:

```bash
python data_scraping_code.py \
    --topic "Black Lives Matter" \
    --start_date 2020-06-01 \
    --end_date 2020-06-30 \
    --output tweets.csv \
    --limit 200 \
    --replies_limit 50 \
    --headless \
    --english_only
```

The scraper stores information such as:

- Tweet ID
- Author
- Conversation ID
- Parent/replied-to tweet
- Timestamp
- Text
- Likes
- Retweets/shares
- Reply count

---

# 2. NLP Preprocessing

### `02_preprocessing_and_visulaization.ipynb`

Raw Twitter/X text is transformed into structured features before graph construction and prediction.

### Text preprocessing

The pipeline performs:

- Lowercasing
- URL removal
- Special-character removal
- Stopword removal
- Duplicate removal
- Empty-text filtering

### NLP features

The project extracts:

- **Sentiment:** TextBlob polarity scores
- **Named entities:** spaCy `en_core_web_sm`
- **Token representations:** BERT tokenizer using `bert-base-uncased`

The processed data is stored in CSV format for subsequent analysis.

---

# 3. Toxicity and Spike Analysis

### `03_spike_analysis.py`

The `ImprovedHybridSpikeAnalyzer` enriches the dataset with toxicity, engagement, contextual, and temporal features.

## Hate/Toxicity Signal

The spike analyzer computes an **enhanced hate score** using multiple signals, including:

- Hate/toxicity-related keywords
- Toxicity amplifiers
- Emotional intensifiers
- Capitalization intensity
- Exclamation-mark intensity

The resulting score is a **heuristic/model-derived signal**, not a human-annotated ground-truth hate-speech label.

## Virality

A normalized virality score is calculated from engagement:

```text
Virality ≈ likes + 3 × shares + 2 × comments
```

The score is normalized to a 0–1 range.

## Spike Detection

Posts are aggregated over time and analyzed using multiple signals:

- Sentiment
- Enhanced hate score
- Post volume
- Virality
- Engagement

The implementation compares these signals against rolling historical statistics to identify periods exhibiting unusually high activity or toxicity.

## Content Categorization

The analyzer categorizes posts using contextual keyword groups, including:

- Politics
- Immigration
- Protest
- Terrorism
- Crime
- Economy
- International events
- Social issues

## External Context

Detected spike periods can be enriched using **The Guardian Open Platform API** to examine relevant external news context.

This provides contextual information around detected activity; it should not be interpreted as proof that a particular news event caused a spike.

---

# 4. Temporal Graph Modeling

### `04_tgn_prediction.ipynb`

The project represents Twitter/X conversations as directed interaction graphs.

### Graph representation

**Nodes**

Each node represents a tweet/reply and can contain features such as:

- Sentiment
- Engagement
- Hate intensity
- Temporal information
- Centrality-related features

**Edges**

Edges capture conversational relationships such as:

- Replies
- Mentions
- Retweets
- Conversation connections

This representation allows the project to incorporate **conversation structure alongside textual features**.

---

# 5. Future Tweet Generation

The prediction pipeline uses transformer-based language generation to simulate possible future conversational turns.

### Main approach

**GPT-2 + TGN-derived contextual information**

Historical conversation context is used to generate a synthetic continuation representing a possible future tweet.

### Baseline

**DistilGPT-2 + ETHOS**

The baseline uses DistilGPT-2 without the additional TGN context.

The generated tweet is subsequently passed through the toxicity classifier so that its predicted toxicity can be compared with the toxicity of an actual future tweet.

---

# 6. Toxicity Classification

The project uses:

**`s-nlp/roberta_toxicity_classifier`**

to estimate toxicity for:

1. Generated future tweets
2. Actual future tweets from later points in the conversation

The comparison provides an experimental measure of how well the generated continuation reflects the toxicity of the observed future conversation.

---

# Experimental Results

The repository contains a **more recent experimental run** than the results reported in the original paper.

The current results below therefore represent the **updated repository experiment**, while the paper contains results from an earlier run.

### Updated Results

| Model | MAE ↓ | RMSE ↓ | Accuracy ↑ | F1 ↑ |
|-------|------:|--------:|-----------:|-----:|
| **TGN + GPT-2 + ETHOS** | 0.234 | 0.484 | 76.6% | **0.750** |
| DistilGPT-2 + ETHOS | 0.234 | 0.484 | 76.6% | 0.683 |
| GPT-2 + DeHateBERT | **0.128** | **0.357** | **87.2%** | **0.847** |

> **Important:** These results are from an updated experimental run and are not intended to represent a definitive benchmark. The dataset and evaluation set are relatively small, and the project is primarily intended to demonstrate the feasibility of combining temporal graph context with transformer-based toxicity analysis.

### Interpretation

The experiments show that transformer-based toxicity prediction can be combined with conversational graph features in a single research pipeline.

However, the results **do not establish that TGN consistently outperforms simpler approaches**. Performance varies across the evaluated configurations, and the limited dataset prevents statistically strong conclusions.

The main contribution of this project is therefore the **architecture and experimental framework for context-aware toxicity analysis**, rather than claiming state-of-the-art predictive performance.

---

# Dataset

The repository contains a final concatenated dataset with approximately **3,800 Twitter/X posts and replies** collected from politically and socially sensitive discussions.

The dataset contains features spanning:

### Tweet-level information

- `id`
- `conversation_id`
- `reply_to_id`
- `reply_to_nm`
- `speaker_nm`
- `timestamp`
- `text`

### Engagement

- `likes`
- `shares`
- `num_comments`

### NLP

- `cleaned_text`
- `sentiment`
- `named_entities`
- `tokens`

### Toxicity and propagation

- `enhanced_hate_score`
- `hate_intensity`
- `virality_score`

### Temporal/contextual features

- `spike_day`
- `spike_cause`
- `spike_external_cause`
- `content_category`
- `contextual_relevance`

### Graph features

- `user_influence`
- `conversation_centrality`
- `reply_chain_depth`

> The repository dataset represents the broader project artifact. The experimental dataset size used in the earlier paper evaluation was smaller; therefore, repository dataset size and paper evaluation size should not be treated as interchangeable.

---

# Technologies

### Data Collection

- Python
- Selenium
- Chrome WebDriver
- WebDriver Manager
- BeautifulSoup
- Requests
- `langdetect`

### NLP

- spaCy
- TextBlob
- NLTK
- BERT tokenizer
- Transformers
- PyTorch

### Generative AI

- GPT-2
- DistilGPT-2

### Toxicity Detection

- RoBERTa-based toxicity classifier
- ETHOS-oriented toxicity analysis
- DeHateBERT

### Graph Modeling

- NetworkX
- Temporal Graph Network concepts
- Reply/conversation graphs
- Graph centrality features

### Data Analysis & Visualization

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- WordCloud

### External Data

- The Guardian Open Platform API

---

# Installation

Install the required Python packages:

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

Download the required NLTK resources:

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
```

---

# Running the Project

## 1. Collect Twitter/X Data

Run:

```bash
python data_scraping_code.py \
    --topic "Black Lives Matter" \
    --start_date 2020-06-01 \
    --end_date 2020-06-30 \
    --output tweets.csv \
    --limit 200 \
    --replies_limit 50 \
    --headless \
    --english_only
```

## 2. Preprocess the Data

Open:

```text
pre-processing_and_visulaization.ipynb
```

Run the notebook to generate:

- Cleaned text
- Sentiment scores
- Named entities
- BERT tokens
- Exploratory visualizations

## 3. Run Spike Analysis

Use:

```text
Spike_Tgn.py
```

to generate toxicity, virality, contextual, and temporal features.

## 4. Run Graph Construction and Prediction

Open:

```text
Tgn+Pred.ipynb
```

The notebook performs the graph-based modeling and future toxicity prediction experiments.

---

# Limitations

This project is intentionally presented as a **proof of concept** rather than a production-ready moderation system.

### Limited Dataset

The experimental dataset is relatively small, limiting statistical confidence and generalization across different communities, topics, languages, and conversational styles.

### Temporal Evaluation

Although the architecture incorporates temporal graph modeling, the current evaluation does not provide a comprehensive large-scale assessment of temporal forecasting performance.

### Heuristic Toxicity Signals

Some spike-analysis features are derived from keyword and rule-based signals rather than human-annotated moderation labels.

### Model-Generated Text

GPT-2 generates synthetic future conversational turns. Generated text may not accurately reproduce the linguistic or semantic characteristics of an actual future response.

### Interpretability

The current implementation exposes graph-derived attributes such as centrality and toxicity scores, but does not provide a dedicated moderator-facing explanation or visualization system.

### Real-Time Deployment

The sequential pipeline involving graph construction, text generation, and toxicity classification introduces computational latency and is not designed for real-time production moderation.

### Generalization

The dataset primarily represents a limited set of politically and socially sensitive conversations. Larger and more diverse datasets are required to evaluate generalization.

### Platform Dependency

The data collection pipeline depends on Twitter/X page structure and Selenium-based scraping, which may break when platform interfaces or access mechanisms change.

---

# Future Work

Potential extensions include:

- Evaluation on substantially larger datasets
- More rigorous temporal forecasting experiments
- Human-annotated toxicity labels
- Multilingual toxicity analysis
- More sophisticated temporal graph architectures
- Explainable graph-based predictions
- Moderator-facing visualizations
- Improved external-event attribution
- Real-time or streaming graph processing
- Evaluation across a wider range of communities and conversational domains

---

# Research Context

This repository accompanies the research work:

**"Conversational Toxicity Prediction: A TGN-Augmented Transformer Approach"**

The work explores the combination of:

```text
Language
   +
Conversation Structure
   +
Temporal Information
   +
External Context
   ↓
Context-Aware Toxicity Analysis
```

The project is intended to demonstrate the **feasibility of incorporating temporal and conversational structure into transformer-based toxicity analysis**, and to provide a modular foundation for future research on larger datasets.

---

# Disclaimer

This repository is intended for **academic research and experimentation only**.

The toxicity and hate-intensity scores produced by the system are model-derived or heuristic signals and should not be treated as definitive moderation labels.

The system is **not intended for automated enforcement, high-impact decision-making, or production moderation** without substantially larger datasets, stronger validation, human evaluation, and appropriate safeguards.

---

## Publication

**Conversational Toxicity Prediction: A TGN-Augmented Transformer Approach**

Presented at **ICMLT 2026**.

The repository contains an updated experimental run; therefore, the performance figures shown above may differ from those reported in the original paper.