import numpy as np 
import pandas as pd 
from datetime import date, timedelta, datetime
import time
import csv

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
start_date = date(2020, 4, 12)
end_date = date(2020, 7, 13) #end date, datetime.date(datetime.now()) (this second option is dynamic and changes by date but depends on timezone)

state_names = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
states = pd.get_dummies(state_names)
#print(states)

articles = pd.DataFrame()
for date in daterange(start_date, end_date):
    #print(date)
    words="finalTweets"+date.strftime("%B%d").lower()+".csv"
    df = pd.read_csv(words)
    df['date'] = date
    articles = pd.concat([articles, df],ignore_index=True)
    #for state in df['location'].unique():

state = pd.get_dummies(articles['location'])
articles = pd.concat([articles, state], axis = 1)
articles = articles.drop(columns = ['Unnamed: 0'])
#articlesFile = "articles.csv"
#articles.to_csv(articlesFile)

#print(articles.head())
#print (articles.tail(10))
#print (articles.columns)
#print(articles.head(150))


import sklearn
from sklearn import feature_extraction

 # Generate bag of words object with maximum vocab size of 1000
counter = sklearn.feature_extraction.text.CountVectorizer(max_features = 1000)
# Generate tf-idf object with maximum vocab size of 1000
tf_counter = sklearn.feature_extraction.text.TfidfVectorizer(max_features = 1000)

#for date in daterange(start_date, end_date):
# Get bag of words model as sparse matrix
bag_of_words = counter.fit_transform(articles['tweet'].apply(lambda x: np.str_(articles['tweet'])))
#print(bag_of_words)

# Get tf-idf matrix as sparse matrix
tfidf = tf_counter.fit(articles['tweet'].apply(lambda x: np.str_(articles['tweet'])))
df = articles


#print(df['location'])

#for date in daterange(start_date, end_date):
'''
words="finalTweets"+date.strftime("%B%d").lower()+".csv"
df = pd.read_csv(words)
df = df.drop(columns = ['Unnamed: 0'])
'''
#df['tfidf'] = tf_counter.transform(df['tweet'])
#df[['location','tfidf']].groupby('location').mean()
#print(df[['location','tfidf']].groupby('location').mean())
#print(df.dtypes)
totalData = pd.DataFrame()
for state in df['location'].unique():
    data = []
    for date in daterange(start_date, end_date):
        thisTweet="finalTweets"+date.strftime("%B%d").lower()+".csv"
        twee = pd.read_csv(thisTweet)
        #print(twee.columns)
        #print(twee.head(10))
        if twee['location'].str.contains(state).any():
            avgtfidf = df[(df['location']==state) & (df['date'] == date)].apply(lambda x: tf_counter.transform([x['tweet']]), axis = 1).sum()
            avgtfidf = avgtfidf/len(df[(df['location']==state) & (df['date'] == date)])
            #print(type(avgtfidf))
            #print(avgtfidf)
            #print(state)
            feature = avgtfidf.toarray()
            #df = np.append(feature,[value])
            data.append(feature)
            #break
            #print (data)
    totalData[state] = data
    break

# Get the words corresponding to the vocab index
tf_counter.get_feature_names()
#print (tf_counter.get_feature_names())

tfidfConverted = "tfidf.csv"
totalData.to_csv(tfidfConverted)