# ============================================
# src/nlp_engine.py
# Phase 4: NLP Engine
#
# Purpose: Text preprocessing, sentiment 
#          analysis, keyword extraction
# Called by: Streamlit app, notebooks
# ============================================

import re
import ssl
import nltk
import pandas as pd
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# Fix SSL for Windows
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Download required NLTK data silently
nltk.download('punkt',                    quiet=True)
nltk.download('punkt_tab',               quiet=True)
nltk.download('stopwords',               quiet=True)
nltk.download('wordnet',                 quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def clean_text(text):
    """Remove URLs, special characters, lowercase."""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def preprocess_text(text):
    """Full NLP pipeline: clean → tokenize → stopwords → lemmatize."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    cleaned    = clean_text(text)
    tokens     = word_tokenize(cleaned)
    filtered   = [w for w in tokens
                  if w not in stop_words and len(w) > 2]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]

    return lemmatized


def get_sentiment(text):
    """Returns polarity score and sentiment label."""
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity

    if polarity > 0.1:
        label = 'positive'
    elif polarity < -0.1:
        label = 'negative'
    else:
        label = 'neutral'

    return polarity, label


def run_nlp_pipeline(df):
    """
    Master function: runs full NLP pipeline on DataFrame.
    Returns enriched DataFrame with NLP columns added.
    """
    # Work only on normal messages
    normal_df  = df[df['message_type'] == 'normal'].copy()

    # Step 1: Text preprocessing
    normal_df['processed_tokens'] = normal_df['message'].apply(
        preprocess_text
    )
    normal_df['processed_text']   = normal_df['processed_tokens'].apply(
        lambda x: ' '.join(x)
    )

    # Step 2: Sentiment analysis
    normal_df['polarity']        = normal_df['message'].apply(
        lambda x: get_sentiment(x)[0]
    )
    normal_df['sentiment_label'] = normal_df['message'].apply(
        lambda x: get_sentiment(x)[1]
    )

    return normal_df


def get_sentiment_stats(normal_df):
    """Returns sentiment distribution and per-user sentiment."""
    sentiment_dist = normal_df['sentiment_label'].value_counts()
    user_sentiment = normal_df.groupby('sender')['polarity'].mean().round(3)
    user_sentiment = user_sentiment.sort_values(ascending=False)

    return {
        'sentiment_distribution' : sentiment_dist,
        'user_sentiment'         : user_sentiment,
        'overall_polarity'       : round(normal_df['polarity'].mean(), 3)
    }


def get_tfidf_keywords(normal_df, max_features=20, top_n=5):
    """Returns top TF-IDF keywords per user."""
    user_docs = normal_df.groupby('sender')['processed_text'].apply(
        lambda x: ' '.join(x)
    ).reset_index()
    user_docs.columns = ['sender', 'document']

    tfidf         = TfidfVectorizer(max_features=max_features)
    tfidf_matrix  = tfidf.fit_transform(user_docs['document'])
    feature_names = tfidf.get_feature_names_out()

    keywords = {}
    for i, sender in enumerate(user_docs['sender']):
        scores      = tfidf_matrix[i].toarray()[0]
        top_indices = scores.argsort()[::-1][:top_n]
        keywords[sender] = [
            (feature_names[j], round(scores[j], 3))
            for j in top_indices if scores[j] > 0
        ]

    return keywords