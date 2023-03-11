import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import regex as re
import nltk
import pickle
import DataPreProcess

train_data = DataPreProcess.train_df

# countV = CountVectorizer()
# x = countV.fit_transform(train_data['text'])
# y = train_data['label']

tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True)
x = tfidf_ngram.fit_transform(train_data['text'])
y = train_data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=24, shuffle =True)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

train_scores = []
test_scores = []


def algorithm(model):
    # Training model
    model.fit(x_train, y_train)

    # score of train set
    train_model_score = model.score(x_train, y_train)
    train_scores.append(round(train_model_score, 2))
    y_pred_train = model.predict(x_train)

    # score of test set
    test_model_score = model.score(x_test, y_test)
    test_scores.append(round(test_model_score, 2))
    y_pred_test = model.predict(x_test)

    model_file = 'ensemble_model.sav'

    final_model = Pipeline([
        ('tfidf', tfidf_ngram),
        ('ensemble_model', model)
    ])
    pickle.dump(final_model, open(model_file, 'wb'))

    # Printing results
    print("Train score :", round(train_model_score, 2))
    print("Test score :", round(test_model_score, 2))

    print("\n----------------------Confusion Matrix---------------------- \n")
    print(confusion_matrix(y_test, y_pred_test))


xgb=XGBClassifier(eval_metric='mlogloss', use_label_encoder =False)
# algorithm(xgb)

lr = LogisticRegression(max_iter=120)
# algorithm(lr)

rf = RandomForestClassifier(n_estimators = 100,verbose=3,n_jobs=-1)
# algorithm(rf)

ensemble_model = VotingClassifier(estimators=[('xgb', xgb), ('lr', lr), ('rf', rf)], voting='soft')
algorithm(ensemble_model)