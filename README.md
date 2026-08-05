## Movie Review Sentiment Analysis: Classical NLP & Machine Learning

This project involves text preprocessing, feature engineering, and machine learning modeling for sentiment classification using the IMDB 50K movie reviews dataset. The objective is to classify reviews as positive or negative using classical ML techniques — no deep learning.

📌 See [Live_stream.md](Live_stream.md) to try the app live.

### Files

* `code_sentimental(4).ipynb` — Jupyter Notebook containing the entire workflow:

  * Data loading
  * Exploratory Data Analysis (EDA)
  * Text preprocessing (cleaning, stopword removal, stemming)
  * Feature extraction (BoW, TF-IDF, n-grams)
  * Model training and evaluation

* `movie_data_part1.xls`, `movie_data_part2.xls`, `movie_data_part3.xls`, `movie_data_part4.xls` — The dataset used for analysis (expected to be in the same directory).
* `app.py` — Streamlit web app for live sentiment predictions.
* `sentiment_model.pkl`, `tfidf_vectorizer.pkl` — Saved trained model and vectorizer.

---

### Features

* **EDA** to understand class balance, review length distribution, and text formatting issues.
* **Text preprocessing** including HTML/punctuation removal, negation-aware stopword removal, and stemming.
* **Feature engineering** using Bag-of-Words, TF-IDF, and TF-IDF with bigrams.
* **Model training** using classical ML classifiers:

  * Multinomial Naive Bayes
  * Logistic Regression
  * Hyperparameter tuning using GridSearchCV
* **Model evaluation** using accuracy, F1-score, and confusion matrices.
* **Model interpretability** via top predictive words/phrases for each class.

---

### Results

The notebook evaluates multiple model and feature combinations, comparing them on accuracy and F1-score. The best-performing combination — TF-IDF (unigrams + bigrams) with Logistic Regression — achieved ~90.012% accuracy and a 0.90 F1-score.

---

### Running the app locally

```
pip install -r requirements.txt
streamlit run app.py
```

---

📌 Author

**Ananya Paul**
