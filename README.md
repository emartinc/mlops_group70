# mlops_group70

### a. Overall Goal of the Project
The goal of this project is to operationalize an end-to-end machine learning pipeline for MBTI Personality Type Prediction, classifying individuals into one of 16 personality types based on their written text. Beyond simple prediction, the project focuses on the MLOps lifecycle by building a robust serving API with real-time text processing, implementing a CI/CD pipeline for automated testing and deployment, and establishing continuous monitoring to detect both data drift (evolving language patterns, vocabulary changes) and concept drift (shifting personality expression online).

### b. Dataset
Data Source: Kaggle MBTI Personality Type Dataset https://www.kaggle.com/datasets/datasnaek/mbti-type.
Description: The dataset contains forum posts from PersonalityCafe users with self-reported MBTI types. Features include approximately 50 concatenated posts per person (separated by "|||"), resulting in long text sequences containing various linguistic patterns, vocabulary usage, and writing styles that correlate with personality dimensions across four axes: I/E (Introversion/Extraversion), N/S (Intuition/Sensing), T/F (Thinking/Feeling), and J/P (Judging/Perceiving).
Size: Approximately 8,675 samples. This size is chosen to allow for rapid prototyping of the infrastructure and pipeline without high computational overhead.
Modality: Unstructured text data (natural language posts). Contains severe class imbalance with Introverts (~75%) dominating Extraverts (~25%) and Intuitives (~85%) dominating Sensors (~15%).

### c. Models
Baseline Model: Logistic Regression on TF-IDF features. A simple, interpretable model to establish a performance benchmark and validate preprocessing pipeline.
Production Model: DistilBERT (or RoBERTa-base). A transformer-based language model chosen for its ability to capture complex linguistic patterns and contextual understanding from pre-training. We will implement four independent binary classifiers (one per MBTI dimension) using Hugging Face Transformers, with class-weighted loss and SMOTE to handle severe imbalance. The imbalanced-learn library will be used as an additional open-source tool for addressing class imbalance challenges specific to this dataset.
