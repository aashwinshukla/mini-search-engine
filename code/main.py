import re
from collections import Counter
import string
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


def preprocess(text):
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

    spell = SpellChecker()
    tokens = [spell.correction(word) or word for word in tokens]

    # 9. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 10. POS tag + Lemmatize
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return tokens


with open('catalog.json', 'r') as file:
    corpus = json.load(file)

while True:
    print("==========Welcome user to our Mini Search Engine==========")

    user_input = input("Enter whatever you want to search from our database: ")

    processed_tokens = preprocess(user_input)
    print("Processed query tokens:", processed_tokens)
    break
