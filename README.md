# mlops_group70

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
