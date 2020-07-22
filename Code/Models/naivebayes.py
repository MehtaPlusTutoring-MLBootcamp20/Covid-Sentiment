import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

meow = pd.read_csv("articles2.csv")

meow.loc[(meow["sentiment"] < -1/10) & (meow["sentiment"] >= -1) , "sentiment"] = -1
meow.loc[(meow["sentiment"] < 1/10) & (meow["sentiment"] >= -1/10) , "sentiment"] = 0
meow.loc[(meow["sentiment"] <= 1) & (meow["sentiment"] >= 1/10) , "sentiment"] = 1

#print(meow['sentiment'].unique())
#print(meow['sentiment'].isin([-1]).sum())
#print(meow['sentiment'].isin([0]).sum())
#print(meow['sentiment'].isin([1]).sum())

positive_df = meow[meow['sentiment'] == 1]
train_positive = positive_df[:3000]
test_positive = positive_df[3000:3750]

neutral_df = meow[meow['sentiment'] == 0]
train_neutral = neutral_df[:3000]
test_neutral = neutral_df[3000:3750]

neg_df = meow[meow['sentiment'] == -1]
train_neg = neg_df[:3000]
test_neg = neg_df[3000:3750]

traindf = train_positive.append([train_neutral,train_neg])
testdf = test_positive.append([test_neutral,test_neg])

df = traindf.append(testdf, sort=False)

#print(meow['sentiment'].head(25))
y = df[["sentiment","location","date","Confirmed"]]
X = df[["tweet"]]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidfX = vectorizer.fit_transform(X["tweet"])
#print(vectorizer.get_feature_names())
print(tfidfX.shape)

X_train, X_test, y_train, y_test = train_test_split(tfidfX, y, test_size=0.2, random_state=0)

y_train_sent = y_train[["sentiment"]]
y_test_sent = y_test[["sentiment"]]

import time
from sklearn import svm
from sklearn.metrics import classification_report
import nltk
# Train the classifier
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train,y_train_sent)

from sklearn import metrics
predicted = MNB.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, y_test_sent)
print(str('{:04.2f}'.format(accuracy_score*100))+'%')

#print('positive: ', report[1])
#print('neutral: ', report[0])
#print('negative: ', report[-1])
#----------------------------------------------------------------------------
#import pickle
# pickling the vectorizer
#pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))

# pickling the model
#pickle.dump(classifier_linear, open('classifier.sav', 'wb'))
#-------