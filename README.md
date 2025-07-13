# AI-Customer-Support-Agent-Churn-Prediction-Explainability-with-RAG

## Overview

This is a full-cycle AI assistant prototype for customer support. It predicts customer churn using a structured, class-based pipeline (OOP approach) and explains predictions using SHAP. These explanations are integrated with a RAG (Retrieval-Augmented Generation) system to provide agents with contextual, document-backed answers.

## Features

- Build a modular OOP-based pipeline for churn prediction (XGBoost, PyTorch).

- Use SHAP for local and global model explainability.

- Convert SHAP values into human-readable text summaries.

- Integrate with RAG (LangChain, FAISS, LlamaIndex) to return relevant help articles or actions.

- Set the foundation for an AI-powered customer agent.

## Tech Stack

- Python (Object-Oriented Programming)
- Scikit-learn
- XGBoost
- SHAP
- Pandas / NumPy / Matplotlib / Seaborn
- PyTorch (optional model)
- Jupyter Notebook
- LangChain / RAG tools (chromadb, FAISS)
- LangChain Agent

  
## Project Structure

├── Model.py                 # Core OOP model pipeline
├── Preprocessed_Data.csv   # Input dataset
├── Processed_Data.csv      # Exported merged features + churn label
├── SHAP_Explanations/      # Visualizations and per-row SHAP text
├── README.md               # This file

## Status
  
  Component - Progress

- Data Preprocessing (OOP) - Completed

- Model Training (XGB, Pytorch) - Completed

- SHAP Integration - Completed

- SHAP → Text Conversion - In progress

- RAG Pipeline - In Progress

- Chat UI / API - Planned


## Workflow Overview

1. **Data Loading**: Reads the customer churn dataset.
2. **Preprocessing**:
    - Drops irrelevant columns
    - Scales numeric values
    - Handles categorical variables
3. **Train/Test Split**: Stratified split on churn label
4. **Model Training**: Uses `XGBClassifier`, `Pytorch Neural Net`(Optional)
5. **Prediction & Evaluation**: Accuracy, F1, Recall
6. **Explainability**:
    - SHAP values generated per prediction
    - Visual and textual explanations prepared
7. **RAG Output**:
    - SHAP + prediction explanations are exported per row
    - Used in a downstream agent for answering internal support queries

## OOP Project Architecture

Unlike typical script-based notebooks, this project uses an Object-Oriented Design to promote reusability, maintainability, and scalability.

### `model.py` Key Components

| Method             | Purpose                                                            |
|--------------------|--------------------------------------------------------------------|
| `load_data()`      | Reads and sets internal dataset                                    |
| `set_data()`       | Splits data into `X/y` and handles ID columns                      |
| `split_data()`     | Uses `train_test_split` with stratification for balanced class splits |
| `scale_data()`     | Standardizes selected numerical features                           |
| `train_model()`    | Trains model (e.g., XGBoost)                                        |
| `make_predictions()`| Predicts on test set                                               |
| `track_performance()`| Computes Accuracy, F1, and Recall metrics                        |
| `explain_model()`  | Computes SHAP values and prepares visualizations                   |

This modular structure lets you easily swap models, update preprocessing, or plug in new explainability tools with minimal code changes.

## RAG-Ready: From Prediction to Help

The final SHAP explanation for each prediction is fed into a RAG system to return answers such as:

“This customer is likely to churn because of high MonthlyCharges and lack of OnlineSecurity. Suggesting retention offer or escalation to Tier 2.”

## SHAP Explainability

- Summary Plot

- Waterfall Plot per instance

- Looping over rows to extract SHAP → Natural Language summaries

- Feature contribution tracking

- SHAP text output usable by downstream RAG pipeline

## Ideal For

- AI Portfolio Project

- ML Engineer Take-Home Project

- Job Interviews in AI/ML + LLM

- AI Assistants for B2B / Support Teams

## Let’s Connect

I'm actively seeking AI/ML roles and would love to share more about this project.







