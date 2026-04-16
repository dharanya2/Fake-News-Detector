# Fake News Detector
A machine learning-based NLP project that detects fake news using Natural Language Processing techniques and the Naive Bayes classifier.

## Overview
This project implements a fake news detection system using CountVectorizer for text feature extraction and Multinomial Naive Bayes for classification. The model learns to distinguish between real and fake news based on textual patterns and vocabulary.

## Features
Text Vectorization: Converts raw text data into numerical features using CountVectorizer
Naive Bayes Classification: Uses Multinomial Naive Bayes for binary classification (Real/Fake)
Model Training & Evaluation: Trains on a dataset with 70-30 train-test split
Interactive Prediction: Allows users to test the model with custom news text input
Accuracy Metrics: Evaluates model performance using accuracy score

## Technologies Used
Python - Programming language
Pandas - Data manipulation and CSV handling
Scikit-learn - Machine learning library
CountVectorizer - Text feature extraction
MultinomialNB - Classification algorithm
train_test_split - Data splitting utility
accuracy_score - Model evaluation metric

## How It Works
Data Loading: Reads news data from CSV file
Vectorization: Converts text into a numerical feature matrix using word counts
Train-Test Split: Divides data into training (70%) and testing (30%) sets
Model Training: Trains Multinomial Naive Bayes on training data
Evaluation: Calculates accuracy on test set
Prediction: Accepts user input and predicts whether news is real or fake

## Model Details
Algorithm: Multinomial Naive Bayes
Feature Extraction: CountVectorizer (Bag of Words)
Train-Test Ratio: 70-30 split
Random State: 42 (for reproducibility)
## Future Improvements
Add cross-validation for better model evaluation
Implement additional features (TF-IDF, word embeddings)
Support for multiclass classification
Model persistence (save/load trained models)
Web interface for easier interaction
Add confidence scores for predictions
