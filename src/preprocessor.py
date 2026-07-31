import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from bs4 import BeautifulSoup
import contractions
import emoji
from spellchecker import SpellChecker

nltk.download('punkt',                        quiet=True)
nltk.download('punkt_tab',                    quiet=True)
nltk.download('stopwords',                    quiet=True)
nltk.download('wordnet',                      quiet=True)
nltk.download('averaged_perceptron_tagger',   quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def _get_wordnet_pos(treebank_tag):
    """Map a Penn Treebank POS tag to the WordNet equivalent."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess(text, spell_correct=False):
    """
    Clean and normalise text into a list of lemmatised tokens.

    Parameters
    ----------
    text : str
        Raw input text (user query or document content).
    spell_correct : bool
        Set True only for user queries — never for corpus documents,
        as it can corrupt valid technical terms.

    Returns
    -------
    list[str]
        Processed token list ready for indexing or comparison.
    """
    # 1. Expand contractions  (don't -> do not)
    text = contractions.fix(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Convert emojis to text  (🔥 -> :fire:)
    text = emoji.demojize(text)

    # 4. Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # 5. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 6. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 7. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 8. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Tokenize
    tokens = word_tokenize(text)

    # 10. Spell correction (user queries only)
    if spell_correct:
        spell = SpellChecker()
        tokens = [spell.correction(word) or word for word in tokens]

    # 11. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 12. POS-aware lemmatization
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in pos_tags]

    return tokens
