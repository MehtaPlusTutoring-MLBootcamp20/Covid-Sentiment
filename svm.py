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
# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(X_train, y_train_sent)
t1 = time.time()
prediction_linear = classifier_linear.predict(X_train)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(y_train_sent, prediction_linear, labels=[1, 0, -1], output_dict=True)
print(report)
#print('positive: ', report[1])
#print('neutral: ', report[0])
#print('negative: ', report[-1])
#----------------------------------------------------------------------------
#import pickle
# pickling the vectorizer
#pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))

# pickling the model
#pickle.dump(classifier_linear, open('classifier.sav', 'wb'))
#----------------------------------------------------------------------------

import matplotlib.pyplot as plt 
import datetime

dates = y_train['date'].unique()

converted_dates = list(map(datetime.datetime.strptime, dates, len(dates)*['%Y-%m-%d']))
#y_axis = y_train['Confirmed']
state = 'Texas'
stateX = X_train.toarray()[(y_train['location']==state)] #& (y_train['date']< '2020-04-19')]
#ylist = classifier_linear.predict(stateX)
#print(ylist)
from matplotlib import pyplot as plt



y_df = y_train[(y_train['location']==state)] #& (y_train['date']< '2020-04-19')]
y_df['date'] = pd.to_datetime(y_df['date'])
y_df['predicted_sent'] = classifier_linear.predict(stateX)
y_df = y_df.groupby('date').mean()
print(type(y_df))
plt.scatter(y_df['date'],y_df['predicted_sent'])
plt.yticks(rotation=90)
plt.show()

'''
import matplotlib.pyplot as plt 
from datetime import date, timedelta, datetime


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
start_date = date(2020, 5, 14)
end_date = date(2020, 7, 13)


ylist = []
averagesent = []
for datess in daterange(start_date, end_date):
    after = datess + timedelta(1)
    y_axis = y_train['Confirmed']
    state = 'Texas'
    stateX = X_train.toarray()[(y_train['location']==state) & (y_train['date'] == (datess.strftime("%Y-%m%-d")))]
    ylist = classifier_linear.predict(stateX)
    print(ylist)
    y_df = y_train[(y_train['location']==state)] #& (y_train['date']< '2020-04-19')]
    y_df['date'] = pd.to_datetime(y_df['date'])
    average = sum(ylist)/len(ylist)
    averagesent.append(average)
    plt.scatter(datess.strftime("%Y-%m%-d"), average)
print(averagesent)
plt.scatter(y_df['date'],averagesent)
plt.yticks(rotation=90)
plt.show()'''