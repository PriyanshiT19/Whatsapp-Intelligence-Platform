# AI-Powered WhatsApp Conversation Intelligence Platform

A data science project that analyzes WhatsApp chat exports
and generates meaningful insights using NLP, Machine Learning,
and Interactive Visualizations.

## Features
- Chat parsing and data extraction
- Feature engineering and preprocessing
- Exploratory Data Analysis
- NLP: Sentiment Analysis, Topic Modeling
- ML: Activity Prediction
- Interactive Streamlit Dashboard

## Project Structure

```
whatsapp-intelligence-platform/
├── src/
│   ├── parser.py          # WhatsApp chat parser
│   ├── preprocessor.py    # Feature engineering
│   ├── analyzer.py        # EDA and statistics
│   ├── visualizer.py      # Chart generation
│   ├── nlp_engine.py      # NLP features
│   ├── intelligence.py    # Communication insights
│   └── ml_engine.py       # ML models
├── notebooks/             # Exploration notebooks
├── data/sample/           # Sample chat data
├── app/                   # Streamlit application
└── requirements.txt
```

## Tech Stack
- Python 3.11
- Pandas, NumPy
- NLTK, TextBlob
- Scikit-learn
- Plotly, Matplotlib
- Streamlit

## Setup

```bash
# Clone the repo
git clone https://github.com/PriyanshiT19/Whatsapp-Intelligence-Platform.git

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
```

## Status
🚧 In Development — Phase 1 Complete (Parser + Preprocessor)