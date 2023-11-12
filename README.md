# Natural Language Processing Project - Spam Classification



## Introduction

This project aims to to cover a full NLP pipeline, from the exploratory data analysis (EDA) to the inference phase, on the spam mail classification. The dataset comes from Kaggle, contains around 5000 mails.  
You can find the dataset [here](https://www.kaggle.com/datasets/chandramoulinaidu/spam-classification-for-basic-nlp).


This project is part of my fifth year at EPF Engineering School. It summarizes what I learned during my NLP class in October 2023.

## Installation

1. Clone the NLP_Project repository:
```sh
git clone https://github.com/QuentinAbn/NLP_Project
```

2. Change to the project directory:
```sh
cd NLP_Project
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```


## Repository Composition

This project is composed of 3 different notebooks, a Python file, and a CSV file containing the dataset.

1. **Exploratory Data Analysis Notebook**

   In the "exploratory_analysis" notebook, I first explored the dataset to understand it. Then I tried to spot relevant information about the dataset by making some plots such as a plot about the mails length.

2. **Utils Python File**

   This Python file, called `utils.py`, contains functions that are used in the `baseline_model` and the `deep_learning` notebooks.  
   It contains a preprocessing script that can be changed at any moment. So far, it tokenizes the data, removes punctuation and converts to lowercase, removes stopwords, and lemmatizes the data. 
   It also contains a Class called Model that contains multiple usefull function for the machine learning algorithms used in the `baseline_model` notebook.
   Finally it contains a funtion used to tokenize the data in the `deep_learning` notebook.

3. **Baseline Model Notebook**

   In this second notebook, named `baseline_model` I apply the preprocessing pipeline to our data so we have usable data to use machine learning models. Then I  train a machine learning model without any particular parameter tuning or feature engineering. The goal here is simply to obtain a baseline model which I'll use as reference for future experiments.  
   Then I try to find the best combination of vectorizer/model to resolve the classification problem using a grid search.  
   Once I have this combination, I make another grid search to find the best values for the hyperparameters of the model. Thanks to that I improve the results of the model.


4. **Deep Learning Notebook**

   The aim of this notebook is to show the performance of a Neural Network model on this same classification problem.
   To achieve this, I preprocess the data, build an NN model, train it and finally plot a confusion matrix and some metrics about its performance.

## Acknowledgments

- Ryan Pegoud for the nice and clear lessons, which were an excellent introduction to NLP.
- Kaggle for the dataset availability


## Sources

1. Tensorflow and Keras Documentation for Deep Learning
   - [Tensorflow Documentation](https://www.tensorflow.org/)
   - [Keras Documentation](https://keras.io/)

2. Scikit-learn Documentation for Machine Learning Models
   - [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

3. ChatGPT for encoding and information on GridSearch
   - [ChatGPT](https://www.openai.com/blog/chatgpt)

4. Stack Overflow for debugging
   - [Stack Overflow](https://stackoverflow.com/)


