import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

meow = pd.read_csv("articles2.csv")

y = meow[["sentiment","location","date","Confirmed"]]
X = meow[["tweet"]]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidfX = vectorizer.fit_transform(X["tweet"])
print(vectorizer.get_feature_names())
print(tfidfX.shape)

X_train, X_test, y_train, y_test = train_test_split(tfidfX, y, test_size=0.2, random_state=0)

y_train_sent = y_train[["sentiment"]]
y_test_sent = y_test[["sentiment"]]

#C=[0.1,0.25,0.5,1.]
#penalty=['l1','l2']

#regressor = LinearRegression()
#regressor.fit(X_train,y_train)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train,y_train_sent)
print(reg.score(X_train,y_train_sent))
print(reg.coef_)
print(reg.intercept_)

print(X_train.shape)

predicted_Y_test_sent = reg.predict(X_test)
print(predicted_Y_test_sent)

from sklearn.metrics import max_error
print(max_error(predicted_Y_test_sent, y_test_sent))

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(predicted_Y_test_sent, y_test_sent))
#print(reg.predict(np))
'''
for i in C:
    for j in penalty:
        clf = LogisticRegression(C = i, penalty = j, solver = 'liblinear', max_iter = 100).fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        print(test_score)
            #print("TRAIN:", train_index, "TEST:", test_index)
            #X_train, X_test = X[train_index], X[test_index]
            #y_train, y_test = y[train_index], y[test_index]
            #clf.fit(X_train, y_train)

test_predictions = clf.predict_proba(X_test).T[1]
fpr,tpr,thresholds = metrics.roc_curve(y_test, test_predictions, pos_label = 1)
metrics.auc(fpr,tpr)
'''
#plt.plot(scoring)
#plt.plot(accuracyScore)
#plt.show()