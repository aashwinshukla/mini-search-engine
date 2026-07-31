import re
from collections import Counter
import string
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import contractions
import emoji
from spellchecker import SpellChecker
import json

def clean_text(text):
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


corpus = []
with open('catalog.json', 'r')as file:
    corpus = json.load(file)

while running:
    print("==========Welcome user to our Mini Search Engine==========")

    user_input = []
    user_input = input("Enter whatever you want to search from our database: ")

    cleaned_user_input = [clean_text(doc) for doc in user_input ]
    print(f"Cleaned Input : {cleaned_user_input}")


