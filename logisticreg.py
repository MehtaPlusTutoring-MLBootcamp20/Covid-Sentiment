import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import svm
import numpy as np

tfidf = "tfidf4.csv"
df = pd.read_csv(tfidf)

X = df.drop(columns = ['Unnamed: 0', 'location'])
X=X.values
y = df['Confirmed']
y=y.values
print(np.size(X))
print(np.size(y))


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
 
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit()
print(tscv)

C=[0.1,0.25,0.5,1.]
penalty=['l1','l2']
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

for i in C:
    for j in penalty:
        clf = LogisticRegression(C = i, penalty = j, solver = 'saga', max_iter = 100).fit(X_train, y_train)
        scoring = []
        accuracyScore = []
        for train_index, test_index in tscv.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)

            test_predictions = clf.predict_proba(X_test).T[1]
            
            test_score = clf.score(X_test, y_test)
            testAcScore = clf.accuracy_score(X_test, y_test)
            scoring.append(test_score)
            accuracyScore.append(testAcScore)

            #prediction = clf.predict()
            print(test_score)
            print('i: ', i)
            print('j: ', j)

        plt.plot(scoring)
        plt.plot(accuracyScore)
        plt.show()

'''
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)

# and plot the result
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X.ravel(), y, color='black', zorder=20)
X_test = np.linspace(-5, 10, 300)

loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, color='red', linewidth=3)

from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)  
text_classifier.fit(X_train, y_train)
 
 
predictions = text_classifier.predict(X_test)
 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))
'''