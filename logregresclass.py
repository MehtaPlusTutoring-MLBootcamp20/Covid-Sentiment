import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

meow = pd.read_csv("articles2.csv")

meow.loc[(meow["sentiment"] < -1/10) & (meow["sentiment"] >= -1) , "sentiment"] = -1
meow.loc[(meow["sentiment"] < 1/10) & (meow["sentiment"] >= -1/10) , "sentiment"] = 0
meow.loc[(meow["sentiment"] <= 1) & (meow["sentiment"] >= 1/10) , "sentiment"] = 1

print(meow['sentiment'].unique())
print(meow['sentiment'].isin([-1]).sum())
print(meow['sentiment'].isin([0]).sum())
print(meow['sentiment'].isin([1]).sum())

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
'''
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C = 0.1, penalty = 'l1', solver = 'saga', max_iter = 100).fit(X_train, y_train_sent)
test_score = clf.score(X_test, y_test_sent)
print(test_score)
'''


C=[0.1,0.25,0.5,1.0,5.0,10.0,20.0]
penalty=['l1','l2']
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

for i in C:
    for j in penalty:
        clf = LogisticRegression(C = i, penalty=j, solver = 'saga', max_iter = 200).fit(X_train, y_train_sent)
        test_score = clf.score(X_test, y_test_sent)
        print(test_score)
        print(i, j)
            #print("TRAIN:", train_index, "TEST:", test_index)
            #X_train, X_test = X[train_index], X[test_index]
            #y_train, y_test = y[train_index], y[test_index]
            #clf.fit(X_train, y_train)

'''
clf = LogisticRegression(C = 0.5, penalty='l1', solver = 'saga', max_iter = 200).fit(X_train, y_train_sent)
test_score = clf.score(X_test, y_test_sent)
print(test_score)
#print(i, j)
 '''       
test_predictions = clf.predict(X_test)
print(confusion_matrix(y_test_sent,test_predictions))  
print(classification_report(y_test_sent,test_predictions))  
print(accuracy_score(y_test_sent, test_predictions))
#fpr,tpr,thresholds = metrics.roc_curve(y_test, test_predictions, pos_label = 1)
#metrics.auc(fpr,tpr)

#X_train[y_train['location']=='Florida']
#import pickle
#pickle.dump(clf,open("models.pckle",'wb'))
#y_train['predicted_sentiment'] = X_train.apply(lambda x: clf.predict(x['tweet']))

dates = y_train['date'].unique()
y_train['sentiment']

#ylist = map(lambda x: clf.predict(tfidfX), X_train['tweet'][(X_train['location']=='Florida') & (X_train['date']< '2020-04-19')])
import datetime
converted_dates = list(map(datetime.datetime.strptime, dates, len(dates)*['%Y-%m-%d']))
y_axis = y_train['Confirmed']
floridaX = X_train.toarray()[(y_train['location']=='Texas')] #& (y_train['date']< '2020-04-19')]
#print(floridaX)
ylist = clf.predict(floridaX)
print(ylist)
#ylist.to_csv("predictedSentimentTexas")
from matplotlib import pyplot as plt
y_df = y_train[(y_train['location']=='Texas')] #& (y_train['date']< '2020-04-19')]
y_df['date'] = pd.to_datetime(y_df['date'])
plt.plot(y_df['date'],ylist)
plt.yticks(rotation=90)
plt.show()