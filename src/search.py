import math
from collections import Counter


def build_index(corpus):
    """
    Compute TF-IDF scores for every term in every document.

    Parameters
    ----------
    corpus : list[dict]
        Each document must have a 'tokens' key (list of preprocessed tokens).

    Returns
    -------
    tfidf_corpus : list[dict]
        One dict per document mapping term -> TF-IDF score.
    doc_freq : Counter
        Number of documents each term appears in.
    """
    N = len(corpus)

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
            idf = math.log((N + 1) / (doc_freq[term] + 1)) + 1   # smoothed IDF
            scores[term] = tf * idf
        tfidf_corpus.append(scores)

    return tfidf_corpus, doc_freq


def rank_results(query_tokens, corpus, tfidf_corpus, doc_freq):
    """
    Score every document against the query and return results sorted
    by relevance (highest score first).

    Parameters
    ----------
    query_tokens : list[str]
        Preprocessed tokens from the user query.
    corpus : list[dict]
        Original corpus documents.
    tfidf_corpus : list[dict]
        TF-IDF scores built by build_index().
    doc_freq : Counter
        Document frequencies built by build_index().

    Returns
    -------
    list[tuple[float, dict]]
        (score, document) pairs, sorted descending by score.
        Only documents with score > 0 are included.
    """
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

    results.sort(key=lambda x: x[0], reverse=True)
    return results
