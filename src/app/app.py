import os
import json

import gdown
import joblib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from keras.src.legacy.preprocessing.text import tokenizer_from_json
from keras.src.utils import pad_sequences

from fastai.learner import load_learner
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


csv_path = os.path.join(BASE_DIR, "przefiltrowanebezprezydentapowinnobycok.csv")
df = pd.read_csv(csv_path)

LANGUAGES = ["pl","en"]

wykresy = [
    "Ilość znaków a sentyment",
    "Ilość wyrazów a sentyment",
    "Najczęściej używane słowa a sentyment",
    "Najczęściej używane słowa a sentyment(sw)",
    "Ilość znaków a sentyment(sw)",
    "Ilość wyrazów a sentyment(sw)",
    "Analiza bigramów i trigramów",
    "Emocjonalny profil tekstu",
    "Róznorodność słownictwa(entropia)",
    "Analiza części mowy",
    "Analiza występowania negacji"
]


ulmfit_path       = os.path.join(BASE_DIR, "best_ulmfit.pkl")
logreg_model_path = os.path.join(BASE_DIR, "logistic_regression_model.joblib")
tfidf_vec_path    = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")
stacking_model_path = os.path.join(BASE_DIR, "stacking_model.pkl")
lstm_model_path   = os.path.join(BASE_DIR, "lstm_best_model.h5")
tokenizer_path    = os.path.join(BASE_DIR, "tokenizer.json")

file_id = "1LUKN_sLV2O0_J1zgnQeyZMd3H7FcFZm1"
drive_url = f"https://drive.google.com/uc?id={file_id}"



if not os.path.exists(ulmfit_path):
    print("Pobieranie modelu ULMFiT z Google Drive...")
    gdown.download(drive_url, ulmfit_path, quiet=False)
    print("Pobieranie zakończone!")
else:
    print("Model ULMFiT już istnieje - pominięto pobieranie.")



# Wczytanie modelu ULMFiT
if not os.path.exists(ulmfit_path):
    print("Model nie istnieje, pobieranie z Google Drive...")
    gdown.download(drive_url, ulmfit_path, quiet=False)
    print("Pobieranie zakończone!")

with open(ulmfit_path, "rb") as f:
    ulmfit_model = load_learner(f)

# Wczytanie pozostałych modeli (LogReg, Stacking, LSTM)
models = {
    "model1": {
        "model": joblib.load(logreg_model_path),
        "vectorizer": joblib.load(tfidf_vec_path)
    },
    "model2": {
        "model": pickle.load(open(stacking_model_path, "rb")),
        "vectorizer": joblib.load(tfidf_vec_path)
    },
    "model3": {
        "model": tf.keras.models.load_model(lstm_model_path),
        "tokenizer": None  # uzupełnimy niżej
    },
    "ulmfit": {
        "model": ulmfit_model
    }
}

# Wczytanie tokenizera dla modelu LSTM
with open(tokenizer_path, "r", encoding="utf-8") as f:
    tokenizer_json_data = json.load(f)
models["model3"]["tokenizer"] = tokenizer_from_json(tokenizer_json_data)

# Mapowanie etykiet numerycznych na tekst
labels = {0: "Negatywny", 2: "Neutralny", 4: "Pozytywny"}



@app.route("/")
def home():
    lang = request.args.get("lang","pl")
    if lang not in LANGUAGES:
        return redirect(url_for("index", lang="pl"))
    return render_template(f"index_{lang}.html", lang=lang)


@app.route("/analizy")
def analizy():
    lang = request.args.get("lang","pl")
    if lang not in LANGUAGES:
        return redirect(url_for("analizy", lang="pl"))
    return render_template(f"analizy_{lang}.html", lang=lang, wykresy=wykresy)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    lang = request.args.get("lang", "pl")
    if lang not in LANGUAGES:
        return redirect(url_for("predict", lang="pl"))

    sentiment = None
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        model_name = request.form.get("model", "")

        if not text or model_name not in models:
            sentiment = "Błąd: niepoprawne dane"
        else:
            model_data = models[model_name]

            # Modele klasyczne (Logistic Regression, Stacking)
            if "vectorizer" in model_data:
                vectorized = model_data["vectorizer"].transform([text])
                prediction = model_data["model"].predict(vectorized)
                sentiment = labels.get(int(prediction[0]))

            # Model LSTM 
            elif "tokenizer" in model_data:
                tokenizer = model_data["tokenizer"]
                sequences = tokenizer.texts_to_sequences([text])
                padded_sequences = pad_sequences(sequences, maxlen=100)
                preds = model_data["model"].predict(padded_sequences)
                argmax_indices = preds.argmax(axis=1)
                # Mapowanie: 0->0, 1->2, 2->4
                mapped = [0 if i == 0 else 2 if i == 1 else 4 for i in argmax_indices]
                sentiment = labels.get(mapped[0])

            # Model ULMFiT 
            elif model_name == "ulmfit":
                ulmfit_pred = model_data["model"].predict(text)
                # ulmfit_pred[0] to etykieta fast.ai (np. "0", "1", "2"),
                # ale może się zdarzyć, że jest w formie str – zależnie od
                # tego jak definowane są klasy w dls.
                # Jeśli to "0" / "1" / "2", zmapujmy je na 0/2/4:
                # (Jeśli natomiast to nazwa "Negatywny"/"Pozytywny",
                # to wystarczy ustawić sentiment = ulmfit_pred[0])
                ulmfit_label_str = ulmfit_pred[0]
                if ulmfit_label_str.isdigit():
                    ulmfit_num = int(ulmfit_label_str)
                    # 0 -> 0, 1 -> 2, 2 -> 4
                    mapped_label = 0 if ulmfit_num == 0 else 2 if ulmfit_num == 1 else 4
                    sentiment = labels.get(mapped_label)
                else:
                    # W razie gdyby ULMFiT zwracał bezpośrednio słowo "Negatywny"/"Neutralny"/"Pozytywny"
                    # wystarczy:
                    sentiment = ulmfit_label_str

    return render_template(f"predict_{lang}.html", lang=lang, models=models.keys(), sentiment=sentiment)


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8080)


