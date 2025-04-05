# TWITTER-SENTIMENT-ANALYSIS

![Banner](image1.jpeg)



Authors:   
 - [Calvince Kaunda](https://github.com/CalvinceKaunda)   
 - [Samuel Marimbi](https://github.com/S-Marimbi)  
 - [Sharon Paul](https://github.com/Sharzyp)  
 - [Brenda Mutai](https://github.com/Brendamutai)


     
## Overview
Not only does a business launch a new product but also evaluates the performance of the  product. Performance may be in terms of profits or how it is received by the target market. It is necessary to consider customer feedback in order to improve the quality of the product and have better products in general in the market. A useful technique for analysing the reviews of the product is through social media reviews. Businesses can evaluate how customers feel regarding the product via analysis of posts, tweets and threads about the product.

## Business Understanding
Apple would like to review customer sentiments from Twitter about their products. To address this they are interested in a Natural Language Processing machine learning model that will analyse tweets about Apple products in order to monitor brand perception and respond to customer feedback efficiently

The Goal of this project is to build a model that classifies tweets as positive, negative or neutral in order to automate sentiment analysis

## Data Understanding
The dataset used for this project is from Kaggle: [Apple Twitter Sentiment (CrowdFLower)](https://www.kaggle.com/datasets/slythe/apple-twitter-sentiment-crowdflower)
The dataset contains multiple features such as the date of the tweet, confidence of the sentiment, sentiment of the tweet/text, and text/tweets, among others. Our target is the sentiment and we would like to predict the sentiment based on the text/tweet 

## Data Preparation
The dataset was cleaned, relevant features extracted and relevant transformations applied to preprocess the data and ensure it is ready for modelling.
The dataset was split into a train set(80% of original dataset) and a test set(20% of original dataset)

## Modelling
The dataset had imbalanced features and techniques such as SMOTE were applied to balance the class distributions and in turn improving model performance.
Our baseline model was the Logistic Regression Model which was used to evaluate performance at a base level. Additional models such as Decision Trees, Random Forest, XGBoost and Naive Bayes were trained, tested and evaluated on the dataset

## Evaluation 
Logistic Regression model appeared to be the best model in general. It had a balanced performance with a good accuracy, stable F1-score and a high AUC.
In terms of ROC/AUC the logistic regression model also has AUC values above 0.79 indicating the model has a reasonably good ability to distinguish between the three classes (Negative, Neutral and Positive). An AUC of 0.5 would represent random chance and an AUC of 1.0 would indicate perfect classification. Our AUC score is closer to 1.0 indicating the model is doing well in terms of classification.

## Model Tuning
We focused on tuning two models: XGBoost and Logistic Regression.
GridSearchCV was used to identify best parameters which were then used to tune the models.
Both models performed well on the trainig data but on the unseen test data, Logistic Regression achieved a higher AUC than XGBoost indicating that the model is better at classification.
Tuning improved Logistic Regression significantly, boosting test accuracy by 2%.
We then used Logistic Regression for making our predictions.

## Conclusion
Logistic Regression is the best model for sentiment classification and should be used for business applications.
XgBoost showed better precision but lower recall, indicating it was more conservative in classifying positive sentiment.
Overfitting was reduced in XGBoost, however its generalization ability still fell short of Logistic Regression.

## Recommendations and Next Steps
We recommend the Logistic Regression model for business applications 
The model can still be tuned further by considering word embeddings such as BERT instead of TF-IDF. Additionally we can explore n-gram combinations for better phrase detection, hence increasing the accuracy of the model futher.
More labelled data can also be gathered in order to improve generalization and semi-supervised learning can also be used to leverage unlabelled text data.
