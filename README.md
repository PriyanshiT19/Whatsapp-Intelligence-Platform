# 💬 AI-Powered WhatsApp Conversation Intelligence Platform

An end-to-end Data Science & Natural Language Processing (NLP) web platform that parses WhatsApp export `.txt` files and transforms unstructured chat logs into actionable behavioral insights, sentiment analytics, and interactive visual dashboards. Built with **Python 3.11**, **Pandas**, **NLTK**, **TextBlob**, **Scikit-Learn**, **Plotly**, and deployed as a **Streamlit** web app.

---

## 🌟 Key Features

- 📂 **Robust Chat Parsing Engine**: Uses custom Regular Expressions (Regex) to handle 12-hour (AM/PM) and 24-hour time formats, multi-country date formats (`MM/DD/YY` and `DD/MM/YY`), multi-line message continuations, URLs, emojis, media, deleted messages, and system notifications.
- ⚙️ **Feature Engineering Pipeline**: Transforms raw text into 14 analytical features including `datetime`, `hour`, `weekday`, `word_count`, `message_length`, `emoji_count`, `url_count`, `is_media`, and `is_deleted`.
- 📊 **Exploratory Data Analytics (EDA)**: Computes user engagement metrics, peak activity hours and days, message share percentages, and communication frequency.
- 🧠 **NLP & Sentiment Analytics Engine**:
  - **Text Preprocessing**: Custom cleaning, tokenization, NLTK stopword filtering, and WordNet lemmatization.
  - **Sentiment Analysis**: Evaluates per-message and per-user sentiment polarity (-1.0 to +1.0) and subjectivity using TextBlob.
  - **Keyword Extraction**: Identifies unique user-specific conversation topics using TF-IDF (Term Frequency - Inverse Document Frequency).
- 📈 **Interactive Dashboards**: Rendered using Plotly (interactive hover/zoom bar & pie charts) and Matplotlib (WordClouds) inside a clean multi-tab Streamlit dashboard.

---

## 🛠️ Tech Stack & Libraries

| Domain | Tech / Library |
|---|---|
| **Language & Environment** | Python 3.11.4, VS Code, Virtual Environment (`.venv`) |
| **Data Pipelines & EDA** | Pandas 2.1.0, NumPy 1.24.3 |
| **Natural Language Processing** | NLTK 3.8.1, TextBlob 0.17.1, Scikit-Learn 1.3.0 |
| **Interactive Visualizations** | Plotly 5.16.1, Matplotlib 3.7.2, WordCloud 1.9.2 |
| **Web Framework & UI** | Streamlit 1.27.0 |
| **Utility Libraries** | Emoji 2.8.0, URLExtract 1.8.0 |

---

## 📁 Project Architecture

```text
AI-Powered WhatsApp Conversation Intelligence Platform/
├── app/
│   ├── main.py              # Streamlit Web Dashboard Application
│   └── pages/               # Modular sub-pages (for future expansion)
├── data/
│   ├── raw/                 # Uploaded/raw chat files
│   ├── processed/           # Feature-engineered processed datasets
│   └── sample/              # Sample chat export for testing
├── notebooks/
│   ├── 01_parsing_exploration.ipynb   # Regex & parsing development
│   ├── 02_eda_exploration.ipynb       # Exploratory statistical analysis
│   └── 03_nlp_exploration.ipynb       # Sentiment analysis & TF-IDF exploration
├── src/
│   ├── parser.py            # Regex WhatsApp chat parsing module
│   ├── preprocessor.py      # Feature engineering pipeline
│   ├── analyzer.py          # Summary statistics & aggregation logic
│   ├── visualizer.py        # Interactive Plotly chart generator
│   └── nlp_engine.py        # Text preprocessing, Sentiment & TF-IDF module
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation


## Setup

```bash
# Clone the repo
git clone https://github.com/PriyanshiT19/Whatsapp-Intelligence-Platform.git

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
```

## How to use
1. Open WhatsApp on your phone or desktop.
2. Open any chat → Tap Settings/More → Export Chat → Select Without Media.
3. Launch the platform locally and upload your exported .txt file via the sidebar uploader.
4. Interactively explore the Overview, Visualizations, and NLP Insights tabs!