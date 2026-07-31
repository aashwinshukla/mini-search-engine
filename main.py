import json
import os

from src.preprocessor import preprocess
from src.search import build_index, rank_results

# ── Load and index corpus ────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'catalog.json')

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    corpus = json.load(f)

for doc in corpus:
    doc['tokens'] = preprocess(doc['title'] + ' ' + doc['content'])

tfidf_corpus, doc_freq = build_index(corpus)

# ── Menu ─────────────────────────────────────────────────────────────────────

search_history = []

while True:
    print("\n========================")
    print("   Mini Search Engine")
    print("========================")
    print("1. Search")
    print("2. Search History")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == '1':
        user_input = input("\nEnter your search query: ").strip()
        if not user_input:
            print("\nQuery cannot be empty.")
            continue

        query_tokens = preprocess(user_input, spell_correct=True)
        print(f"\nProcessed tokens : {query_tokens}")

        results = rank_results(query_tokens, corpus, tfidf_corpus, doc_freq)
        search_history.append(user_input)

        if not results:
            print("\nNo results found.")
        else:
            top = results[:5]
            print(f"\nFound in {len(results)} document(s) — showing top {len(top)}:\n")
            for pos, (score, doc) in enumerate(top, start=1):
                print(f"  {pos}. [{doc['type'].upper()}] {doc['title']}")
                print(f"     URL   : {doc['url']}")
                print(f"     Score : {score:.4f}\n")

    elif choice == '2':
        if not search_history:
            print("\nNo search history yet.")
        else:
            print("\n--- Search History ---")
            for i, query in enumerate(search_history, start=1):
                print(f"  {i}. {query}")

    elif choice == '3':
        print("\nGoodbye!")
        break

    else:
        print("\nInvalid choice. Please enter 1, 2, or 3.")
