# mlops_group70


## Project 1. Credit Risk Assessment

### a. Overall Goal of the Project

The goal of this project is to operationalize an end-to-end machine learning pipeline for Credit Risk Assessment, classifying loan applicants as likely to default or not. Beyond simple prediction, the project focuses on the MLOps lifecycle by building a robust serving API with real-time data validation, implementing a CI/CD pipeline for automated testing and deployment, and establishing continuous monitoring to detect both data drift (demographic changes) and concept drift (economic shifts).

### b. Dataset
Data Source: German Credit Risk Dataset https://www.kaggle.com/datasets/uciml/german-credit.

Description: The dataset classifies people described by a set of attributes as good or bad credit risks. Features include financial status (checking account status, credit history), demographic info (age, employment), and loan details (purpose, amount).

Size: Approximately 1,000 samples. This size is chosen to allow for rapid prototyping of the infrastructure and pipeline without high computational overhead.

Modality: Structured tabular data (categorical and numerical features).

### c. Models
Baseline Model: Logistic Regression. A simple, interpretable linear model to establish a performance benchmark.

Production Model: Random Forest Classifier (or XGBoost). A robust ensemble method chosen for its ability to handle non-linear relationships and mix of categorical/numerical data without extensive preprocessing.


## Project 2. Movie Recommendation System

### a. Overall Goal of the Project

The goal of this project is to operationalize an **end-to-end movie recommendation system** using the MovieLens 20M dataset, capable of generating personalized movie recommendations for users based on their historical ratings. Rather than focusing on maximizing model performance, the project emphasizes **production-oriented machine learning practices**, including a clear data processing pipeline, reproducible model training, offline evaluation with ranking metrics, and deployment of a recommendation service that can be queried by downstream applications.


### b. Dataset

**Data Source:** MovieLens 20M Dataset (GroupLens). https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset

**Description:**
The dataset contains explicit user–movie interactions in the form of ratings ranging from 0.5 to 5 stars. Each record includes a `userId`, `movieId`, `rating`, and `timestamp`, with additional metadata available for movies such as title and genres. The data supports collaborative filtering approaches and realistic recommendation workflows.

**Size:**
Approximately 20 million ratings from tens of thousands of users and movies. For practical experimentation, the pipeline supports sampling or filtered splits while remaining compatible with the full dataset.

**Modality:**
Structured interaction data with categorical identifiers (users and movies) and optional side information, represented through embeddings.


### c. Models

**Baseline Model:** Matrix Factorization.
A classical collaborative filtering approach where users and movies are represented by latent embedding vectors, and relevance is estimated via their dot product. This model provides a simple and fast benchmark.

**Production Model:** Neural Collaborative Filtering (simplified NeuMF).
A lightweight neural recommendation model that combines user and movie embeddings and feeds them into a small multi-layer perceptron to capture non-linear user–item interactions. The model is trained for rating prediction or top-N recommendation and evaluated using ranking metrics such as Hit@K and NDCG@K, with a focus on deployability and inference efficiency.
