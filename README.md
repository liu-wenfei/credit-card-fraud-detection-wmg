# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A comparative study of machine learning and deep learning approaches for credit card fraud detection on an imbalanced dataset.

## Overview

This project explores fraud detection using the public ULB Credit Card Fraud Dataset. It compares several models and techniques, including:

- Logistic Regression
- Feed-forward Neural Network
- Autoencoder
- Isolation Forest
- W1B semi-supervised enhancement
- ENS ensemble method

The focus is not only on model accuracy, but also on handling class imbalance, temporal drift, model interpretability, and practical deployment considerations.

## Project Structure

- [credit_card_fraud_final.ipynb](credit_card_fraud_final.ipynb)  
  Main notebook containing data loading, exploratory analysis, preprocessing, model training, evaluation, and visualization.

- [fraud_demo.html](fraud_demo.html)  
  An interactive demo page for visualizing the fraud detection workflow in a browser.

## Key Highlights

- Implements both supervised and unsupervised fraud detection methods
- Evaluates models using PR-AUC, F1 score, recall, and calibration-related analysis
- Investigates concept drift over time and its impact on model performance
- Uses SHAP-based interpretability analysis for model explanation

## Requirements

Python 3.10+ is required.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the notebook

```bash
jupyter notebook
```

Open [credit_card_fraud_final.ipynb](credit_card_fraud_final.ipynb) and run the cells in order.

### 2. Open the demo

Open [fraud_demo.html](fraud_demo.html) directly in a browser to view the demo interface.

## Dataset

This project uses the publicly available ULB Credit Card Fraud Detection dataset, which is characterized by:

- severe class imbalance
- temporal transaction patterns
- a realistic fraud detection setting

## Notes

This project is intended for research and educational purposes, especially for exploring:

- imbalanced classification
- deep learning vs traditional machine learning
- anomaly detection
- model interpretability and evaluation
