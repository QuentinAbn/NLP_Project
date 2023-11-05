# Natural Language Processing Project - Spam Classification

I didn't finish this project yet !

## Introduction

This project aims to to cover a full NLP pipeline, from the exploratory data analysis (EDA) to the inference phase, on the spam mail classification. The dataset comes from Kaggle, contains around 5000 mails.

This project is part of my fifth year at EPF Engineering School. It summarizes what I learned during my NLP class in October 2023.

## Repository Composition

This project is composed of 2 different notebooks, a Python file, and a CSV file containing the dataset.

1. **Exploratory Data Analysis Notebook**

   In the "exploratory_analysis" notebook, I first explored the dataset to understand it. Then I tried to spot relevant information about the dataset by making some plots such as a plot about the mail length.

2. **Utils Python File**

   This Python file, called `utils.py`, contains a preprocessing script that can be changed at any moment. So far, it tokenizes the data, removes punctuation and converts to lowercase, removes stopwords, and lemmatizes the data.  
   It also contains another function used to produce reports on the results of machine learning models, this function is used further in the project in the "baseline_model" notebook.

3. **Baseline Model Notebook**

   In this second notebook, named "baseline_model" we apply the preprocessing pipeline to our data so we have usable data to use machine learning models. Then we  train a machine learning model without any particular parameter tuning or feature engineering. The goal here is simply to obtain a baseline model which we'll use as reference for future experiments.  
   So far I couldn't make it further in the project. The next steps are improving this model and then use a deep learning techniques on the dataset.


4. **Deep Learning Notebook**

   To be done...

## Acknowledgments

- Ryan Pegoud for the nice and clear lessons, which were an excellent introduction to NLP.
- Kaggle for the dataset availability
