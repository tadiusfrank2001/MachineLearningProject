# Sentiment Analysis Machine Learning Project

## Overview

The goal of this project was to perform sentiment analysis on a Twitter dataset using two different machine learning models: Decision Tree and Logistic Regression. 
The focus was on evaluating which classification model performed better in terms of accuracy.

## Sentiment Classification
Sentiment classification is a type of text classification problem, where instead of classifying based on the topic of the text, the focus is on the sentiment or opinion lexicon that indicates whether an opinion is positive or negative. 
To reduce the complexity and improve classification accuracy, I filter out neutral lexicons, or “stopwords,” using Python’s `NLTK` library, as they are less informative. 
This approach allows us to treat sentiment classification as a binary classification problem, ignoring the neutral class and classifying text as either positive or negative.

After vectorizing the text and counting the frequency of sentiment lexicons, I fitted a Logistic Regression model on the preprocessed data with a 70-30 train-test split. 
I then evaluated whether removing neutral lexicons impacts the results and implement a Decision Tree classifier to compare performance.

## Challenges and Adjustments
Initially, the project code was structured to handle non-NLP-based datasets, which only involved numerical data. 
To address this, I augmented the code to properly handle the text data by reading the CSV file, removing stop words, and filtering out characters such as emojis and "@" mentions. 
Additionally, the dataset was contextually labeled based on a specific scenario (e.g., a video game Twitter thread), where logically negative actions might be labeled as positive.

The dataset consisted of over 1.6 million entries, which posed challenges in terms of runtime and computational resources. To manage this, we trimmed the dataset to a size that our machines could handle efficiently.

## Technologies Used
`Python`: Programming language used.
`scikit-learn`: For implementing the Decision Tree and Logistic Regression models.
`Pandas`: For data manipulation and analysis.
`NumPy`: For numerical computations.
`NLTK`: For natural language processing tasks.

## Project Structure
`experiment.py`: Cleans and preprocesses the Twitter dataset, including removing stop words and extra punctuation.
`classify.py`: Contains the Logistic Regression and Decision Tree classifiers used to train and evaluate the models.
`data/`: Directory containing the Twitter tweet data, split into multiple 25MB files for easier management.

