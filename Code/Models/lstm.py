#INCOMPLETE

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

from collections import Counter
all_text2 = ' '.join(meow)
# create a list of words
words = all_text2.split()
# Count all the words using Counter Method
counts = Counter(words)


vocab = sorted(counts, key=counts.get, reverse=True)
print(vocab)

vocab_to_int = {word:ii for ii, word in enumerate(vocab, 1)}
print (vocab_to_int)

reviews_ints = [] 
for review in reviews_split:    
    reviews_ints.append([vocab_to_int[word] for word in review.split()])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

y_train_sent = y_train[["sentiment"]]
y_test_sent = y_test[["sentiment"]]

