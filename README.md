# Stock Alpha Pipeline — AI-Based Stock Outperformance Prediction Platform

An end-to-end machine learning pipeline designed to identify stocks with a higher probability of outperforming their sector peers over a 6-month investment horizon.

The system combines financial fundamentals, technical indicators, SEC filing sentiment analysis, natural language processing, and machine learning to generate data-driven stock rankings.

The project includes:

- Automated financial data collection pipeline
- Feature engineering framework
- NLP-based SEC filing sentiment analysis
- Machine learning model training and optimization
- Time-series cross-validation
- Out-of-sample backtesting
- Web-based dashboard for prediction visualization and analysis

---

# Project Motivation

Traditional stock analysis requires manually combining financial statements, market trends, and company news. This project explores how machine learning can integrate multiple sources of financial information to identify potential market outperformers.

Instead of predicting exact stock prices, the model performs a **ranking-based prediction task**:

> "Which companies have the highest probability of outperforming their sector over the next 6 months?"

---

# System Pipeline
SEC EDGAR Financial Data
|
v
Fundamental Feature Engineering
|
|
Yahoo Finance OHLCV Data
|
v
Technical Feature Engineering
|
|
SEC 8-K Filings
|
v
NLP Sentiment Extraction
(TF-IDF + SVD + VADER)
|
v
Feature Fusion
|
v
Machine Learning Model
(HistGradientBoosting + Optuna)
|
v
Predictions + Backtesting
|
v
Interactive Web Dashboard

---

# Data Sources

## SEC EDGAR

Quarterly company filings are collected from the SEC EDGAR API.

Extracted financial metrics include:

- Revenue
- Gross profit
- Operating income
- Net income
- Total assets
- Debt
- Equity
- Cash flow
- Shares outstanding


## Yahoo Finance

Daily market data is collected using Yahoo Finance:

- Open
- High
- Low
- Close
- Volume

Historical data is used to generate technical indicators and time-series features.

---

# Feature Engineering

## Fundamental Features

The pipeline calculates company financial health indicators:

- Return on Assets (ROA)
- Return on Equity (ROE)
- Operating Margin
- Debt-to-Equity Ratio
- Debt-to-Assets Ratio
- Current Ratio
- Revenue Growth
- Income Growth
- Asset Growth
- Accrual Ratio


## Technical Market Features

Price-based features include:

- 3-month momentum
- 6-month momentum
- 12-month momentum
- Daily returns
- Rolling volatility
- Momentum composite indicators


## NLP Sentiment Features

SEC 8-K filings are processed to extract company sentiment.

Pipeline:
8-K Filing Text
|
v
HTML Cleaning
|
v
Text Normalization
|
v
VADER Sentiment Analysis
|
v
TF-IDF Vectorization
|
v
Truncated SVD Topic Extraction
|
v
Quarterly Sentiment Features

Generated NLP features:

- Average sentiment score
- Positive filing ratio
- Negative filing ratio
- Sentiment volatility
- Filing frequency
- Latent topic representations

---

# Machine Learning Model

## Model Architecture

The prediction model uses:

**HistGradientBoostingClassifier**

with:

- Optuna hyperparameter optimization
- Time-series cross-validation
- Time-decay sample weighting
- Sector-relative feature engineering


## Engineered Features

Additional financial factors created:

- Momentum reversal
- Quality spread
- Earnings quality
- Leverage-growth interaction
- Composite momentum score
- ROIC proxy
- Sector-relative indicators


---

# Training Methodology

The pipeline avoids future data leakage using chronological validation.

Training:
2010 - 2022

Testing: 
2023 - Present

Evaluation methods:

- TimeSeriesSplit validation
- Out-of-sample testing
- Sector-based performance comparison
- Portfolio-style ranking evaluation

---

# Backtesting

The model is evaluated as an investment ranking system.

Metrics calculated:

- ROC-AUC
- Classification accuracy
- Top-quintile average return
- Bottom-quintile average return
- Long-short performance spread
- Sector-level performance
- Year-by-year performance


Example strategy evaluation:
Rank stocks by prediction probability
    Highest Probability
            |
            v
   Potential Outperformers


    Lowest Probability
            |
            v
   Potential Underperformers
   ---

# Dashboard Application

A web dashboard was developed to showcase model outputs and research findings.

The dashboard provides:

- Stock prediction rankings
- Confidence scores
- Model performance metrics
- Feature importance analysis
- Backtesting results
- Sector comparisons
- Historical prediction analysis

The application is deployed on a server for remote access.

---

# Technology Stack

## Programming

- Python

## Machine Learning

- Scikit-learn
- HistGradientBoosting
- Optuna

## Data Processing

- Pandas
- NumPy
- PostgreSQL

## Financial Data

- SEC EDGAR API
- Yahoo Finance API

## Natural Language Processing

- NLTK VADER
- TF-IDF
- Truncated SVD

## Visualization & Deployment

- Matplotlib
- Seaborn
- Web Dashboard
- Server Deployment

---
---

# Future Improvements

Potential extensions:

- Transformer-based financial language models
- Earnings call transcript analysis
- Macroeconomic indicators
- Portfolio optimization algorithms
- Real-time prediction updates
- Deep learning time-series models


---

# Disclaimer

This project is developed for educational and research purposes.

Machine learning predictions should not be considered financial advice.

---

# Author

**Aditya Srivatsa Adiraju**

Robotics Engineering Student

Machine Learning | Artificial Intelligence | Data-Driven Systems
