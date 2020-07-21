import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

meow = pd.read_csv("articles3.csv")

meow.loc[(meow["sentiment"] < -3/5) & (meow["sentiment"] >= -1) , "sentiment"] = -2
meow.loc[(meow["sentiment"] < -1/5) & (meow["sentiment"] >= -3/5) , "sentiment"] = -1
meow.loc[(meow["sentiment"] < 1/5) & (meow["sentiment"] >= -1/5) , "sentiment"] = 0
meow.loc[(meow["sentiment"] < 3/5) & (meow["sentiment"] >= 1/5) , "sentiment"] = 1
meow.loc[(meow["sentiment"] <= 1) & (meow["sentiment"] >= 3/5) , "sentiment"] = 2

y = meow[["sentiment","location","date","Confirmed"]]
X = meow[["tweet"]]

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


C=[0.1,0.25,0.5,1.]
penalty=['l1','l2']
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

for i in C:
    for j in penalty:
        clf = LogisticRegression(C = i, penalty=j, solver = 'saga', max_iter = 100).fit(X_train, y_train_sent)
        test_score = clf.score(X_test, y_test_sent)
        print(test_score)
        print(i, j)
            #print("TRAIN:", train_index, "TEST:", test_index)
            #X_train, X_test = X[train_index], X[test_index]
            #y_train, y_test = y[train_index], y[test_index]
            #clf.fit(X_train, y_train)

        
test_predictions = clf.predict_proba(X_test).T[1]
print(confusion_matrix(y_test_sent,test_predictions))  
print(classification_report(y_test_sent,test_predictions))  
print(accuracy_score(y_test_sent, test_predictions))
#fpr,tpr,thresholds = metrics.roc_curve(y_test, test_predictions, pos_label = 1)
#metrics.auc(fpr,tpr)