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
from sklearn import metrics
import nltk
# Train the classifier

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
SGDC = SGDClassifier()
LSVC = LinearSVC()

LSVC.fit(X_train, y_train_sent)
accuracy_score_lsvc = metrics.accuracy_score(LSVC.predict(X_test),y_test_sent)
print ('accuracy_score_lsvc = ' + str('{:4.2f}'.format(accuracy_score_lsvc*100))+'%')

SGDC.fit(X_train,y_train_sent)
accuracy_score_sgdc = metrics.accuracy_score(SGDC.predict(X_test),y_test_sent)
print ('accuracy_score_sgdc = ' + str('{:4.2f}'.format(accuracy_score_sgdc*100))+'%')


import matplotlib.pyplot as plt 
import datetime
dates = y_train['date'].unique()
converted_dates = list(map(datetime.datetime.strptime, dates, len(dates)*['%Y-%m-%d']))
#y_axis = y_train['Confirmed']
#state = 'Connecticut'
#stateX = X_train.toarray()[(y_train['location']==state)] #& (y_train['date']< '2020-04-19')]
stateX = X_train.toarray()
#ylist = classifier_linear.predict(stateX)
#print(ylist)
from matplotlib import pyplot as plt
#_df = y_train[(y_train['location']==state)] #& (y_train['date']< '2020-04-19')]
y_df = y_train
y_df['date'] = pd.to_datetime(y_df['date'])
#print(type(y_df['date']))
print(y_df['date'].dtype)
y_df['predicted_sent'] = SGDC.predict(stateX)
# import IPython
# IPython.embed()
y_df = y_df.groupby('date', as_index=False).mean()
# print(type(y_df))
# plt.scatter(y_df.index,y_df['predicted_sent'])
# plt.scatter(y_df['date'],y_df['predicted_sent'])
plt.title("United States Average Sentiment")
plt.xlabel('Date')
plt.ylabel('Sentiment')
plt.scatter(y_df['date'],list(y_df['predicted_sent']))
plt.yticks(rotation=90)
plt.show()