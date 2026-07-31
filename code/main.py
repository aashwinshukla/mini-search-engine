import re
from collections import Counter
import string
import math
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import contractions
import emoji
from spellchecker import SpellChecker
import json

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def get_wordnet_pos(treebank_tag):
    """Convert Penn Treebank POS tag to WordNet POS tag for accurate lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess(text, spell_correct=False):
    # 1. Expand contractions (don't -> do not, I'm -> I am)
    text = contractions.fix(text)

    # 2. Lowercase
    text = text.lower()

    text = emoji.demojize(text)

    # 3. Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # 4. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 5. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 6. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 7. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 8. Tokenize
    tokens = word_tokenize(text)

    # 9. Spell correction (user input only, not corpus)
    if spell_correct:
        spell = SpellChecker()
        tokens = [spell.correction(word) or word for word in tokens]

    # 10. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 11. POS tag + Lemmatize
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return tokens


def compute_tfidf(corpus):
    """Build a TF-IDF index from preprocessed corpus documents."""
    N = len(corpus)

    # Count how many documents each term appears in (for IDF)
    doc_freq = Counter()
    for doc in corpus:
        for term in set(doc['tokens']):
            doc_freq[term] += 1

    tfidf_corpus = []
    for doc in corpus:
        token_count = len(doc['tokens'])
        term_freq = Counter(doc['tokens'])
        scores = {}
        for term, count in term_freq.items():
            tf = count / token_count
            idf = math.log((N + 1) / (doc_freq[term] + 1)) + 1  # smoothed IDF
            scores[term] = tf * idf
        tfidf_corpus.append(scores)

    return tfidf_corpus, doc_freq


def rank(query_tokens, corpus, tfidf_corpus, doc_freq):
    """Score and rank documents against the query using TF-IDF."""
    N = len(corpus)
    results = []

    for i, doc in enumerate(corpus):
        score = 0.0
        for term in query_tokens:
            if term in tfidf_corpus[i]:
                idf = math.log((N + 1) / (doc_freq[term] + 1)) + 1
                score += tfidf_corpus[i][term] * idf
        if score > 0:
            results.append((score, doc))

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)
    return results


with open('catalog.json', 'r') as file:
    corpus = json.load(file)

# Preprocess all corpus documents once upfront
for doc in corpus:
    doc['tokens'] = preprocess(doc['title'] + ' ' + doc['content'])

# Build TF-IDF index once after corpus is preprocessed
tfidf_corpus, doc_freq = compute_tfidf(corpus)

search_history = []

while True:
    print("\n========================")
    print("   Mini Search Engine")
    print("========================")
    print("1. Search")
    print("2. Search History")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ")

    if choice == '1':
        user_input = input("\nEnter your search query: ")

        processed_tokens = preprocess(user_input, spell_correct=True)
        print("Processed query tokens:", processed_tokens)

        results = rank(processed_tokens, corpus, tfidf_corpus, doc_freq)

        search_history.append(user_input)

        if not results:
            print("\nNo results found.")
        else:
            print(f"\nTop {min(5, len(results))} results:\n")
            for rank_pos, (score, doc) in enumerate(results[:5], start=1):
                print(f"{rank_pos}. [{doc['type'].upper()}] {doc['title']}")
                print(f"   URL   : {doc['url']}")
                print(f"   Score : {score:.4f}\n")

    elif choice == '2':
        if not search_history:
            print("\nNo search history yet.")
        else:
            print("\n--- Search History ---")
            for i, query in enumerate(search_history, start=1):
                print(f"{i}. {query}")

    elif choice == '3':
        print("\nGoodbye!")
        break

    else:
        print("\nInvalid choice. Please enter 1, 2, or 3.")
