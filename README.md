# Mini Search Engine

A command-line search engine built in Python that indexes a catalog of documents and returns ranked results using TF-IDF. Includes a full NLP preprocessing pipeline with tokenization, lemmatization, spell correction, and emoji handling.

This is a learning project — built from scratch without any search framework.

---

## Features

- **NLP preprocessing pipeline** — contraction expansion, emoji conversion, HTML stripping, stopword removal, POS-aware lemmatization
- **Spell correction** — typos in queries are automatically corrected
- **TF-IDF ranking** — results ranked by relevance, not just keyword presence
- **Search history** — tracks every query made in the current session
- **Clean menu UI** — simple numbered menu for navigation

---

## Project Structure

```
mini-search-engine/
│
├── main.py                 # Entry point — menu and app logic
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py     # Text cleaning and NLP pipeline
│   └── search.py           # TF-IDF indexing and ranking
│
├── data/
│   └── catalog.json        # Document catalog (25 documents)
│
├── requirements.txt        # Python dependencies
└── README.md
```

---

## How It Works

### 1. Preprocessing

Every document and every query goes through the same pipeline in `src/preprocessor.py`:

1. Expand contractions (`don't` → `do not`)
2. Lowercase
3. Convert emojis to text (🔥 → `:fire:`)
4. Strip HTML tags
5. Remove URLs and numbers
6. Remove punctuation
7. Tokenize
8. Spell correction *(queries only — never run on corpus)*
9. Remove stopwords
10. POS-aware lemmatization (`running` → `run`, `better` → `good`)

### 2. Indexing

`src/search.py` builds a TF-IDF index from the preprocessed corpus once at startup.

- **TF (Term Frequency)** — how often a word appears in a document
- **IDF (Inverse Document Frequency)** — rare words across the corpus score higher than common ones
- Both are multiplied to produce a relevance score per term per document

### 3. Ranking

When you search, your query goes through the same preprocessing pipeline, then each document is scored by summing the TF-IDF weights of all matching query terms. Documents are sorted highest score first.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

**1. Clone the repository**

```bash
git clone https://github.com/your-username/mini-search-engine.git
cd mini-search-engine
```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the program**

```bash
python main.py
```

---

## Usage

```
========================
   Mini Search Engine
========================
1. Search
2. Search History
3. Exit

Enter your choice (1-3):
```

**Search example:**

```
Enter your search query: machine learning algorithms

Processed tokens : ['machine', 'learn', 'algorithm']

Found in 3 document(s) — showing top 3:

  1. [ARTICLE] Introduction to Machine Learning
     URL   : https://example.com/tech/intro-to-ml
     Score : 0.7902

  2. [ARTICLE] Understanding Neural Networks
     URL   : https://example.com/tech/neural-networks-explained
     Score : 0.3201

  3. [TUTORIAL] Python Programming for Beginners
     URL   : https://example.com/tutorials/python-beginners
     Score : 0.1045
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `beautifulsoup4` | Strip HTML tags from text |
| `nltk` | Tokenization, stopwords, POS tagging, lemmatization |
| `contractions` | Expand contractions before cleaning |
| `emoji` | Convert emojis to readable text |
| `pyspellchecker` | Spell correction for user queries |

---

## What I Learned Building This

- How to build an NLP text preprocessing pipeline step by step
- Why the order of preprocessing steps matters
- The difference between stemming and lemmatization, and when to use each
- How TF-IDF works and why it is better than simple word counting for ranking
- How to structure a Python project into modules (`src/`, `data/`)
- Debugging common beginner mistakes like iterating over strings character by character

---

## Possible Next Steps

- Phrase search (`"dynamic memory"` as a single unit, not two separate tokens)
- Statistics screen (total words indexed, total documents)
- Persistent search history saved to a file
- Highlight matched words in results
- Load `.txt` files from a folder instead of a JSON catalog

---

## License

MIT
