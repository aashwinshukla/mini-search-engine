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
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

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

while True:
    print("==========Welcome user to our Mini Search Engine==========")

    user_input = input("Enter whatever you want to search from our database: ")

    cleaned_user_input = clean_text(user_input)
    print(f"Cleaned Input : {cleaned_user_input}")

    tokenized_user_input = word_tokenize(cleaned_user_input)
    print("Tokenized user input: ", tokenized_user_input)

    stop_words = set(stopwords.words('english'))
    filtered_user_input = [word for word in tokenized_user_input if word not in stop_words]
    print("Stopword Removed Input: ", filtered_user_input)

    stemmer = PorterStemmer
    stemmed_user_input = [stemmer.stem(word) for word in filtered_user_input]
    print("Stemmed Input: ", stemmed_user_input)
    break


