# MBTI Personality Classifier

An end-to-end MLOps pipeline for predicting Myers-Briggs personality types from text using deep learning.

## Overview

This project implements a production-ready machine learning system that analyzes writing samples to predict personality types based on the Myers-Briggs Type Indicator (MBTI). Unlike traditional 16-class classification approaches, we use a **multi-task learning** framework with 4 independent binary classifiers for better performance and interpretability.

## Key Features

- **🧠 Multi-task Binary Classification**: 4 independent classifiers for E/I, S/N, T/F, J/P dimensions
- **🤖 Transformer-based Model**: DistilBERT (67M parameters) for contextual understanding
- **📊 Advanced Data Handling**: Random window sampling for sequences >512 tokens
- **🚀 Production-ready Serving**: FastAPI backend with Streamlit UI
- **⚙️ Modular Configuration**: Hydra-based configs for reproducible experiments
- **📦 Data Versioning**: DVC integration with Google Cloud Storage
- **🐳 Containerization**: Docker Compose for seamless deployment
- **✅ Comprehensive Testing**: Unit and integration tests with coverage

## Architecture Highlights

### Model Architecture

```
Input Text (up to 512 tokens)
         ↓
DistilBERT Encoder (pretrained)
         ↓
[CLS] Token Representation
         ↓
Shared Dense Layer (768 → 768)
         ↓
    ┌────┴────┬────┬────┐
    ↓         ↓    ↓    ↓
  E/I       S/N  T/F  J/P
  Head      Head Head Head
    ↓         ↓    ↓    ↓
 Binary  Binary Binary Binary
 Output  Output Output Output
    └────┬────┴────┴────┘
         ↓
   MBTI Type (e.g., INTJ)
```

### System Architecture

```
┌─────────────┐
│   Training  │ (Docker container)
│   Service   │ → Saves model to ./models/
└─────────────┘
       ↓
┌─────────────┐     HTTP        ┌──────────┐
│ FastAPI API │ ←───────────────│ Streamlit│
│  (port 8000)│                 │    UI    │
└─────────────┘                 │(port 8501)│
       ↑                        └──────────┘
       │ Loads model from
   ./models/best.ckpt
```

## Dataset

- **Source**: [Kaggle MBTI Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)
- **Size**: ~8,675 samples from PersonalityCafe forum
- **Format**: Concatenated posts per user with self-reported MBTI type
- **Challenge**: Severe class imbalance (75% Introverts, 85% Intuitives)

## Performance

The model treats MBTI as 4 binary classification tasks:

| Dimension | Task | Metric |
|-----------|------|--------|
| E/I | Extraversion vs Introversion | Binary F1 |
| S/N | Sensing vs Intuition | Binary F1 |
| T/F | Thinking vs Feeling | Binary F1 |
| J/P | Judging vs Perceiving | Binary F1 |

This approach provides:
- Better handling of class imbalance
- More interpretable predictions
- Dimension-level probability scores

## Quick Links

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Architecture Overview](architecture/overview.md)
- [API Reference](api/data.md)

## Tech Stack

- **ML Framework**: PyTorch Lightning
- **Model**: Hugging Face Transformers (DistilBERT)
- **Configuration**: Hydra
- **API**: FastAPI
- **UI**: Streamlit
- **Data Versioning**: DVC + Google Cloud Storage
- **Containerization**: Docker + Docker Compose
- **Testing**: pytest
- **Package Management**: uv

## Project Status

This project is actively maintained and ready for production deployment.

---

Built with ❤️ using [MLOps Template](https://github.com/SkafteNicki/mlops_template)
