# Sentiment-Analysis
## Introduction
Sentiment analysis is a crucial task in natural language processing, aiming to determine the sentiment expressed in a given text. This project's objective is to create a model with high accuracy compared to established models such as BERT, LSTM, SVM, and Logistic Regression. The BERT model produced 89% accuracy, while LSTM, SVM, and Logistic Regression produced 75%, 74.75%, and 65%.

## Methodology

### 2.1 Dataset
The dataset used in this study comprises Indian user tweets collected during the COVID-19 lockdown period. It consists of 3090 tweets focusing on topics related to COVID-19, coronavirus, and lockdown. The data included tweets collected on the dates between 23 March 2020 and 15 July 2020 and the text has been labelled as fear, neutral, and joy.

### 2.2 Data - Preprocessing
The different data preprocessing techniques that are applied to the dataset include the removal of emojis, special characters, and noise. Text data was tokenized, lemmatized, and subjected to sentiment analysis using the VADER sentiment intensity analyzer.

### 2.3 Feature Engineering
Text data was transformed into numerical features using TF-IDF vectorization. The sentiment analysis results were incorporated into the dataset as additional features.

## Model Development

### 3.1 Neural Network Architecture:
For the neural network model, the Multi-Layer Perceptron (MLP) classifier was employed. Utilizing the MLPClassifier from scikit-learn, the model consisted of an input layer, hidden layers with a ReLU activation function, and an output layer with a softmax activation function. The Adam optimizer was employed for training, and the sparse categorical cross-entropy loss function facilitated the model's learning process. The model achieved following evaluation metrics:
- Accuracy: 92.42%
- Precision: 92.34%
- Recall:92.42%
- F1 Score: 92.31%

### 3.2 Decision Tree Classifier:
The Decision Tree classifier, a non-linear model, exhibited commendable performance in sentiment analysis. The model achieved the following metrics:
- Accuracy: 88.62%
- Precision: 88.46%
- Recall: 88.62%
- F1 Score: 88.51%

### 3.3 Naive Bayes Classifier:
The Naive Bayes classifier, known for its simplicity and efficiency, demonstrated competitive results in sentiment analysis. The model achieved the following metrics:
- Accuracy: 82.44%
- Precision: 83.11%
- Recall: 82.44%
- F1 Score: 82.31%

### 3.4 Gradient Boosting Classifier:
The Gradient Boosting classifier, an ensemble model, provided insightful results in sentiment analysis. The model achieved the following metrics:
- Accuracy: 76.84%
- Precision: 81.90%
- Recall: 76.84%
- F1 Score: 74.90%

## Evaluation Metrics
The model's performance was evaluated using standard metrics:
- Accuracy: The proportion of correctly classified instances.
- Precision: The ratio of correctly predicted positive observations to the total predicted positives.
- Recall: The ratio of correctly predicted positive observations to the total actual positives.
- F1 Score: The harmonic mean of precision and recall.

## Conclusion
In conclusion, the developed neural network model has demonstrated superior accuracy compared to established models such as BERT, LSTM, SVM, and Logistic Regression in sentiment analysis.

## Reference
Chintalapudi, N., Battineni, G., & Amenta, F. (2021). Sentimental analysis of COVID-19 tweets using deep learning models. Infectious disease reports, 13(2), 329-339.
