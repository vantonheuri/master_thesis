# Master Thesis - Utilizing Sentiment Analysis to Guide Investment Decisions Across Global Markets

This repository includes all the notebooks and scripts developed for my final thesis project. The objective is to analyze the **impact of financial sentiment** on the stock prices of the “Magnificent Seven” tech companies: Apple, Microsoft, Amazon, Alphabet, Meta, Nvidia, and Tesla.

## Objective

The goal of this thesis is to explore whether financial news sentiment has predictive power over stock price movements. The approach combines:

- LLM-based sentiment scoring
- Econometric models (VAR + IRFs)
- Price and fundamentals data
- Dashboard for visualization

---

## Project Structure

### 1. `1. Data Extraction.ipynb`
Extracts historical prices, fundamentals, and financial news using APIs like Alpha Vantage. This is the starting point of the analysis.

### 2. `2. EDA Initial Study.ipynb`
First **exploratory data analysis**, focused on understanding the raw data, returns, and basic visualizations.

### 3. `2. EDA Utils Study.ipynb`
Another EDA notebook that uses shared utility functions to expand the analysis in a cleaner, more modular way.

### 4. `3. LLM Prompt Engineering.ipynb`
Design and testing of prompts for Large Language Models (LLMs) to evaluate the **sentiment** of news headlines. Used to score how positive or negative each piece of news is.

### 5. `4. VAR IRFs - [TICKER].ipynb`
These notebooks apply **VAR (Vector Autoregression)** and **Impulse Response Functions (IRFs)** to measure how stock prices respond to sentiment shocks. One notebook is created for each company:
- AAPL (Apple)
- MSFT (Microsoft)
- AMZN (Amazon)
- GOOGL (Alphabet)
- META (Meta)
- NVDA (Nvidia)
- TSLA (Tesla)

Each notebook shows the full modeling process and results for one company.

### 6. `5. Initial MVP.ipynb`
This notebook contains the **initial MVP** (Minimum Viable Product) that shaped the core idea of the thesis. It helped to define the methodology and validate the concept.

---

### 7. `utils.py`
A Python script with helper functions for:
- Loading and cleaning data
- Calculating returns
- Integrating sentiment scores

It is used in multiple notebooks to keep the code clean and consistent.

### 8. `stocks_dashboard.py`
Prototype script to create an interactive dashboard using **Streamlit**. The goal is to visualize key results from the analysis.

> Placeholder for the future dashboard preview:
> ![Dashboard Preview](path/to/dashboard_image.png)
