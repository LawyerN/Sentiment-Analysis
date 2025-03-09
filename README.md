# 📊 Twitter Sentiment Analysis with Flask & Machine Learning

![Flask App Screenshot](https://i.imgur.com/71Rs3oK.png)

## 🚀 Overview
This project is a Flask-based web application for analyzing sentiment in tweets using various machine learning models. The application provides **11 different visualizations** representing various aspects of Twitter comments, sentiment distribution, and model performance.
I collected comments in English from various data sets, mainly from kaggle:


https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
https://www.kaggle.com/datasets/kazanova/sentiment140
https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset
https://www.kaggle.com/competitions/tweet-sentiment-extraction
https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset
https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset


The system includes **four sentiment analysis models**. The models are:
- **Logistic Regression**
- **Stacked Model (Naïve Bayes + XGBoost + Logistic Regression)**
- **LSTM (Long Short-Term Memory)**
- **ULMFiT (Universal Language Model Fine-tuning for Text Classification)**

In addition, there are other sentiment classification models in the notebook and files, but they had worse accuracy, so I did not include them in the application(e.g. Random Forest, CNN, 


---

## 🔥 Features
✅ Sentiment classification using multiple ML & Deep Learning models  
✅ **11 visualizations** of Twitter comments and sentiment analysis(some aspects were analysed from stop words (sw means; without stop words, other charts are with them) )
✅ **Comparison of multiple classifiers** in real-time  
✅ RESTful API integration for text-based sentiment analysis  
✅ Web-based interface powered by **Flask**  
✅ bilingual interface Polish and English

---

## 🛠 Technologies Used
- **Backend**: Flask, Python  
- **Frontend**: HTML, CSS, Bootstrap, JavaScript (optional if interactive)  
- **Data Processing**: Pandas, NumPy, Scikit-learn  
- **Visualization**: Seaborn, matplotlib
- **Machine Learning**: Scikit-learn, XGBoost, Naïve Bayes, Logistic Regression  
- **Deep Learning**: TensorFlow/Keras (LSTM), FastAI (ULMFiT)  

---

## 📸 Application Screenshots
🏠 Main Dashboard

![Flask App Screenshot](https://i.imgur.com/jFkV5ls.png)
![Flask App Screenshot](https://i.imgur.com/wf1abnp.png)

📊 Sentiment Analysis
 ![Flask App Ss](https://i.imgur.com/VWVFJvx.png)
 ![Flask App Ss](https://i.imgur.com/PVgkjGO.png)
 ![Flask App Ss](https://i.imgur.com/0M1yipv.png)

---


## 📈 Exploratory Data Analysis (EDA) Visualizations
In addition to standard sentiment distribution plots, we performed **11 in-depth exploratory analyses** to better understand the textual data. These plots illustrate how various textual features relate to sentiment:

1. **Character Count vs. Sentiment**  
   Shows how the number of characters in each tweet correlates with different sentiment classes.

2. **Word Count vs. Sentiment**  
   Analyzes whether the total number of words in a tweet influences its sentiment.

3. **Most Frequently Used Words vs. Sentiment**  
   Highlights the most common words in tweets, grouped by sentiment category.

4. **Most Frequently Used Words vs. Sentiment (Stopwords Removed)**  
   Similar to the above, but excludes common stopwords to reveal more meaningful terms.

5. **Character Count vs. Sentiment (Stopwords Removed)**  
   Compares the character lengths of tweets after removing stopwords, across different sentiment labels.

6. **Word Count vs. Sentiment (Stopwords Removed)**  
   Investigates how the tweet’s word count changes once stopwords are removed, segmented by sentiment.

7. **Bigrams and Trigrams Analysis**  
   Examines the most frequent 2-word and 3-word phrases to find specific linguistic patterns tied to sentiment.

8. **Emotional Profile Analysis**  
   Maps tweets to various emotional categories (like joy, anger, sadness) to see how they align with overall sentiment.

9. **Vocabulary Diversity (Entropy)**  
   Measures the richness of the vocabulary within each sentiment class using an entropy-based metric.

10. **Part-of-Speech Analysis**  
    Shows the distribution of POS tags (nouns, verbs, adjectives, etc.) across tweets with different sentiments.

11. **Negation Usage Analysis**  
    Looks at how negation words (e.g., “not”, “no”) appear in tweets and whether they affect sentiment classification.


---

## 📦 Installation & Setup
Probably you will need to download Microsoft Visual C++ Build Tools and then choose  MSVC v143 - Visual C++ 2019 build tools, Windows 10 SDK, C++ ATL

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/LawyerN/Sentiment-Analysis.git
cd Sentiment-Analysis
cd src
cd app
```


### 🔹 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 🔹 3. Install Dependencies
```bash
python -m pip install --upgrade pip==25.0.1
pip install -r requirements.txt
```

### 🔹 4. Run the Flask Application
```bash
python app.py
```
The application will run on `http://127.0.0.1:8080/`. Then just try it out, choose a model then write a comment and see a result.

---





## 📡 API Endpoints
You can use the RESTful API to analyze sentiment directly:

### 🔹 1. Analyze Text Sentiment
**Endpoint:** `/api/sentiment`  
**Method:** `POST`  
**Payload:**
```json
{
  "text": "This is an amazing project!"
}
```
**Response:**
```json
{
  "sentiment": "Positive"
}




```

---

## 🧪 Model Performance
| Model | Accuracy | Precision | Recall  | F1-score |
|--------|----------|-----------|---------|----------|
| Logistic Regression | 73%      | 74%       | 73%     | 74%      |
| Stacked Model | 74%      | 74%       | 74%      | 74%      |
| LSTM | 89%      | 88%       | 87%     | 88%      |

---
## 📊 Confusion Matrices (Model Performance Analysis)
Confusion matrices help visualize the classification performance of different models. Below are the confusion matrices for each model:

Logistic Regression Confusion Matrix
![Logistic Regression](https://i.imgur.com/ttFFysF.png)


Stacked Model Confusion Matrix
![Stacked Model](https://i.imgur.com/6LwrANe.png)


LSTM Confusion Matrix
![LSTM](https://i.imgur.com/26ZLhOB.png)


These confusion matrices show the number of correct and incorrect predictions for each sentiment category.

---
## 📜 Future Improvements
🔹 Add real-time Twitter scraping & analysis  
🔹 Implement sentiment trend over time  
🔹 Deploy the application on **Heroku/AWS**  

---




## 👨‍💻 Author
**Your Name** – *[Bartosz Gdowski](https://www.linkedin.com/in/bartosz-gdowski-627a0b250/)*  
**GitHub:** *[https://github.com/LawyerN](https://github.com/your-username)*  
🚀 Feel free to contribute or give suggestions!

---

## ⭐ Like This Project?
If you found this project useful, **please give it a star ⭐ on GitHub!**  
