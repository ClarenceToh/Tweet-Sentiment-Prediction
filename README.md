# Tweet-Sentiment-Prediction
Predicting Sentiment of Tweets about the weather

This is part of a kaggle competition "2019 40.016 The Analytics Edge Competition"

Website: https://www.kaggle.com/c/2019tae

## Group Members:
1. Clarence Toh
2. Sharan Sunil
3. Song Zhi Guo

## Project Description
#### Data
All participants are provided a training set containing only two columns with the headers "tweet" and "sentiment."

The "tweet" column contains strings of each tweet. The "sentiment" column refers to three possible sentiments that classify the tweet. A '1' value refers to negative sentiment, '2' refers to neutral sentiment and '3' refers to positive sentiment.

The test dataset contains an "Id" column and a "tweet" column that corresponds to some sentiment which must be predicted.

#### Objective
The aim is to classify as many tweets to the appropriate sentiment. We want the classifier to work well, not only on the data provided but on the private data as well. High accuracy on the public data may not always produce good performance on private data.

#### Approach
##### Naive Models
comp.RMD
1. CART (accuracy: 0.7453)
2. Naive Bayes Classifier (0.9949)
3. Random Forest Classifier (0.9988)
4. Support Vector Machine (0.849)

High accuracy often means an overfitting of data, hence they will be ignored.

##### Bench Mark
all_models.RMD
XGBoost (0.858)

##### Final Model
all_models.RMD
Support Vector Machine (0.855728)

#### Full Description
Further explanation is provided in the PDF report.
