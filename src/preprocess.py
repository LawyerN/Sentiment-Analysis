import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import multiprocessing as mp
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

stop_words = set([
    "the", "and", "is", "in", "to", "of", "a", "that", "this", "it", "on",
    "for", "with", "as", "was", "but", "be", "by", "at", "or", "an", "me","you","have", "my","i","im", "will","it","u","th","lol","rt","get","got","go","work","see","say","like","may"
])



def fast_clean_text(text):
    """Szybsza wersja clean_text bez nltk"""
    if not isinstance(text, str):  # Jeśli nie jest tekstem, zamień na pusty string
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Usunięcie linków
    text = re.sub(r"@\w+|#", "", text)  # Usunięcie @mentions i #
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Usunięcie znaków specjalnych
    words = text.split()  # Zamiast word_tokenize()
    words = [word for word in words if word not in stop_words]  # Usunięcie stop words

    return " ".join(words)

def parallel_apply(df, func, num_cores=mp.cpu_count()):
    df_split = np.array_split(df, num_cores)  # Podział na rdzenie
    pool = mp.Pool(num_cores)  # Tworzymy pulę procesów
    df = pd.concat(pool.map(func, df_split))  # Przetwarzamy równolegle
    pool.close()
    pool.join()
    return df

def process_texts(df):
    df['text'] = df['text'].apply(fast_clean_text)  # Szybsza funkcja

    return df