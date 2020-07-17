import numpy as np 
import pandas as pd 
from datetime import date, timedelta, datetime
import time
import csv
from twarc import Twarc
import nltk  
nltk.download('stopwords')  
from nltk.corpus import stopwords 

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
start_date = date(2020, 4, 12)
end_date = date(2020, 7, 13) #end date, datetime.date(datetime.now()) (this second option is dynamic and changes by date but depends on timezone)

import sklearn

for date in daterange(start_date, end_date):
    words="finalTweets"+date.strftime("%B%d").lower()+".csv"
    df = pd.read_csv(words)
    articles = df["tweet"]
    
    # Generate bag of words object with maximum vocab size of 1000
    counter = sklearn.feature_extraction.text.CountVectorizer(max_features = 1000)
    # Get bag of words model as sparse matrix
    bag_of_words = counter.fit_transform(articles)
    # Generate tf-idf object with maximum vocab size of 1000
    tf_counter = sklearn.feature_extraction.text.TfidfVectorizer(max_features = 1000)
    # Get tf-idf matrix as sparse matrix
    tfidf = tf_counter.fit_transform(articles)
    # Get the words corresponding to the vocab index
    tf_counter.get_feature_names()
    print (tf_counter.get_feature_names())