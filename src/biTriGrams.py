import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\User\PycharmProjects\sentiment\finaldata\przefiltrowanebezprezydentapowinnobycok.csv")

# Funkcja do ekstrakcji n-gramów
def get_top_ngrams(texts, ngram_range=(2, 2), top_n=10):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform(texts)
    ngram_counts = Counter(dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1)))
    return ngram_counts.most_common(top_n)

# Analiza dla każdego sentymentu
for sentiment in df['sentiment'].unique():
    print(f"\nNajczęstsze bigramy dla sentymentu {sentiment}:")
    print(get_top_ngrams(df[df['sentiment'] == sentiment]['text'], (2, 2), 10))

    print(f"\nNajczęstsze trigramy dla sentymentu {sentiment}:")
    print(get_top_ngrams(df[df['sentiment'] == sentiment]['text'], (3, 3), 10))