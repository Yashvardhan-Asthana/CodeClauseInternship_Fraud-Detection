# Fraud Detection System

## Overview
This project is a **Fraud Detection**, an entry-level project assigned to me as part of my Data Science Internship at CodeClause IT Company. The system detects fraudulent transactions using machine learning techniques, leveraging methods such as oversampling, ensemble learning, and feature scaling to build an efficient and accurate classification model.

I would like to express my gratitude to CodeClause IT Company for providing me with this wonderful opportunity to enhance my skills and knowledge during the internship.


## Features
- **Multiple Models**: Logistic Regression.
- **Interactive Predictions**: Accepts user inputs for real-time fraud detection.
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, F1-score, and confusion matrix for all models.

## Prerequisites
- Python 3.8 or higher
- Libraries: `pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`

## Dataset
The dataset is sourced from Kaggle ([link](https://www.kaggle.com/datasets/kartik2112/fraud-detection)) that includes 5,55,720 credit card entries along with the 23 features which involves 'trans_date_trans_time, cc_num, amt, street, city, zip, state, first_name, last_name, lat, long, merchant, merch_lat, merch_long, city_pop, job, dob, trans_num and category, with is_fraud as the target column.

## Workflow
Download Dataset: Searching web and downloaded dataset from Kaggle online data science platform.
Load Dataset: Load and analyze the dataset for missing values and class distribution.
Preprocessing data: Encode categorical features, balance data using SMOTE, and scale numeric features.
Train Model: Train Logistic Regression model.
Evaluate Model: Assess accuracy and other metrics for the model.
Interactive Prediction: Use the trained Voting Classifier to classify transactions.

## Output
The model takes 'amount', 'city', 'merchant name' and 'category' as input from the user and predicts the output as 'Fraud' or 'Non-Fraud'.

## Acknowledgments
Dataset is collected from Kaggle.
This project is assigned by CodeClause IT Company as a part of 'Data Science Internship'. 
