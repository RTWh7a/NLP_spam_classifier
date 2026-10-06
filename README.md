# NLP_spam_classifier
# SMS Spam Classification NLP Project

## Overview

This project focuses on building a machine learning text classification model to accurately distinguish between spam and legitimate (ham) SMS messages. The workflow covers environment setup, data loading, missing value inspection, class distribution evaluation, and exploratory data analysis (EDA).

---

## Prerequisites & Dependencies

To run the project notebook successfully, the following libraries and dependencies are required:

* **Python Packages**: `numpy`, `pandas`, `matplotlib`, `string`, `scikit-learn` (including model selection, classification metrics, feature extraction with `CountVectorizer`, and algorithms like `MultinomialNB` and `SVC`), `nltk`, `cleantext`, `re` (regular expressions), and `wordcloud`.


* **NLTK Dependencies**: `brown`, `names`, `wordnet`, `averaged_perceptron_tagger`, `universal_tagset`, and `stopwords`.



---

## Dataset Details

* **Source**: Loaded from a remote TSV file URL (`[https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)`).


* **Structure**: Formatted as a pandas DataFrame with two primary columns: `label` and `message`.


* **Dataset Shape**: 5,572 rows and 2 columns `(5572, 2)`.


* **Class Distribution**:
* `ham`: 4,825 instances.


* `spam`: 747 instances.





---

## Project Workflow & Steps

### 1. Import Python Packages

* Import all important Python modules required for loading data, text preprocessing, feature extraction, and building text classification models.


* Set random seed to `123` for reproducibility.



### 2. Load the Spam Dataset

* Download necessary NLTK corpora and dependencies programmatically.


* Read the TSV dataset using pandas with tab separation (`sep="\t"`) and assign column names `['label', 'message']`.



### 3. Handle Missing Values

* Check for missing values using pandas' `isna().sum()` method.


* Verified that there are `0` missing values for both `label` and `message` columns.



### 4. Evaluate Class Distribution

* Analyze the balance of the dataset using the `value_counts()` method on the `label` column.



### 5. Exploratory Data Analysis (EDA)

* Implement a custom `collect_words` function to gather, tokenize, and lowercase words separately for spam and ham messages.


* This helps uncover frequent vocabulary patterns used in spam versus legitimate texts.
