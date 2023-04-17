# Automatic Ticket Classification Case Study
A case study to create a model that can automatically classify customer complaints based on the products and services that the ticket mentions.

## Table of Contents
* [General Info](#general-information)
* [Project Contents](#project-contents)
* [Conclusion](#conclusion)
* [Software and Library Versions](#software-and-library-versions)
* [Acknowledgements](#acknowledgements)

### General Information
For a financial company, customer complaints carry a lot of importance, as they are often an indicator of the shortcomings in their products and services. If these complaints are resolved efficiently in time, they can bring down customer dissatisfaction to a minimum and retain them with stronger loyalty. This also gives them an idea of how to continuously improve their services to attract more customers. 

These customer complaints are unstructured text data; so, traditionally, companies need to allocate the task of evaluating and assigning each ticket to the relevant department to multiple support employees. This becomes tedious as the company grows and has a large customer base.

#### Data Set Brief Information
The data set given to you is in the .json format and contains 78,313 customer complaints with 22 features. You need to convert this to a dataframe in order to process the given complaints.

#### Business Objective
As an NLP engineer for a financial company that wants to automate its customer support tickets system. As a financial company, the firm has many products and services such as credit cards, banking and mortgages/loans. 

#### Business Solution
Build a model that is able to classify customer complaints based on the products/services. By doing so, segregate these tickets into their relevant categories and, therefore, help in the quick resolution of the issue.

With the help of non-negative matrix factorization (NMF), an approach under topic modelling, you will detect patterns and recurring words present in each ticket. This can be then used to understand the important features for each cluster of categories. By segregating the clusters, you will be able to identify the topics of the customer complaints. 

Need to do topic modelling on the .json data provided by the company. Since this data is not labelled, need to apply NMF to analyse patterns and classify tickets into the following five clusters based on their products/services:

* Credit card / Prepaid card
* Bank account services
* Theft/Dispute reporting
* Mortgages/loans
* Others 

With the help of topic modelling, we can map each ticket onto its respective department/category. Use this data to train any supervised model such as logistic regression, decision tree or random forest. Using this trained model, classify any new customer complaint support ticket into its relevant department.


### Project Contents
* **Automatic Ticket Classification Case Study** - Jupyter Notebook for Telecom Churn Case Study (Language : Python)
* **complaints-2021-05-14_08_16.json** - JSON file containing dataset
* **README.md** - Readme file


### Conclusion
* Dataset was loaded and was analyzed. Imputation was done and took necessary steps to get clean data.
* Text was preprocessed by using lemmatization. POS tags other than NN was removed.
* Exploratory data analysis (EDA) was performed to get character length of complaints. Unigram, Bigram and Trigram words was found and also word cloud of top 40 words was displayed.
* Converted the raw texts to a matrix of TF-IDF features.
* Non-Negative Matrix Factorization (NMF) was used to find 5 topics.
* Model was built to create the topics for each complaints.
* 4 models were trained - Logistic Regression, Decision Tree, Random Forest and Naive Bayes.
* Logistic Regression with hyperparameter tuning was chosen as final model for inference of complaints.

#### Recommendation
**Logistic Regression with Hyperparameter** model is choosen as the **Best Model** for inference and categorize complaints in to 5 topics.
Model have:
* Accuracy of **94.34** % and F1 score of **94.34** % for train dataset
* Accuracy of **93.36** % and F1 score of **93.35** % for test dataset


### Software and Library Versions
* ![Jupyter Notebook](https://img.shields.io/static/v1?label=Jupyter%20Notebook&message=4.9.2&color=blue&labelColor=grey)

* ![NumPy](https://img.shields.io/static/v1?label=numpy&message=1.21.5&color=blue&labelColor=grey)

* ![Pandas](https://img.shields.io/static/v1?label=pandas&message=1.4.2&color=blue&labelColor=grey)

* ![matplotlib](https://img.shields.io/static/v1?label=matplotlib&message=3.5.1&color=blue&labelColor=grey)

* ![seaborn](https://img.shields.io/static/v1?label=seaborn&message=0.11.2&color=blue&labelColor=grey)

* ![sklearn](https://img.shields.io/static/v1?label=sklearn&message=1.0.2&color=blue&labelColor=grey)

* ![nltk](https://img.shields.io/static/v1?label=nltk&message=3.7&color=blue&labelColor=grey)

* ![spacy](https://img.shields.io/static/v1?label=spacy&message=3.5.0&color=blue&labelColor=grey)


### Acknowledgements
This case study is an assignment, done as part of [Upgrad](https://www.upgrad.com/ ) - **Master of Science in Machine Learning & Artificial Intelligence** programme.


### Contact
Created by [Sanjeev Surendran](https://github.com/Sanjeev-Surendran)


<!-- ## License -->
<!-- This project is not a open source and sharing the project files is prohibited. -->
