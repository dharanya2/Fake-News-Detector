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
