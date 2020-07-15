'''
lowercase
stop words
punctuation
apostrophe
single characters
stemming
lemmatisation
converting numbers
emojis/emoticons
'''
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import pandas as pd
import numpy as np
import csv
import re #regular expression
from textblob import TextBlob
import string
import preprocessor as p
from twarc import Twarc
from datetime import date, timedelta, datetime
import nltk
from nltk.corpus import stopwords
print(stopwords.words('english'))
#from geopy.geocoders import Nominatim
#import countries
'''
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
start_date = date(2020, 5, 14)
end_date = date(2020, 7, 13) #end date, datetime.date(datetime.now()) (this second option is dynamic and changes by date but depends on timezone)

for dates in daterange(start_date, end_date):
    after = dates + timedelta(1)
    #print (after)
    print(after.strftime("%m") + "-" + after.strftime("%d") + "-2020.csv")
    filename=after.strftime("%m") + "-" + after.strftime("%d") + "-2020.csv"
    #change the file name when they are done
'''
data = 'preprocessTest.csv'
def lowercase (textData):
    lowerfile = 'tempLower.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for text in files:
            newFile = np.char.lower(text)
            #print (newFile)
            with open (lowerfile, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow([newFile])
#def stopWords (data):
lowercase(data)
        #break
#files= pd.read_csv(filename)
#newFile= np.char.lower(files)
#print(newFile)