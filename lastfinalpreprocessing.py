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
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import num2words
import demoji

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
start_date = date(2020, 4, 12)
end_date = date(2020, 7, 13)

def preprocessed (textData):
    allfile = 'lastFinishedPreprocess.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for text in files:
            newFile = np.char.lower(text)
            #print("lower: ", newFile)

            newFile = remove_stopwords(str(newFile))
            #print("stop: ", newFile)

            newFile = re.sub('http[s]?://\S+', '', str(newFile))
            #print("URL: ", newFile)

            newFile = np.char.replace(newFile, '\\n', '')
            newFile = np.char.replace(newFile, '\\r', '')

            symbols = "!#$%&()*+-"
            for i in symbols:
                newFile = np.char.replace(newFile, i, ' ')
            symbols = "!./:;<=>?@[\]^_`{|}~"
            for i in range(len(symbols)):
                newFile = np.char.replace(newFile, symbols[i], ' ')
            newFile = np.char.replace(newFile, ',', '')
            newFile = np.char.replace(newFile, "  ", " ")
            #print("punc: ", newFile)
            #print (newFile)
            newFile = np.char.replace(str(newFile), "’", "")

            newFile = re.sub(r"\b[a-zA-Z]\b", "", str(newFile))

            '''ps = PorterStemmer()
            newFile = [ps.stem(word) for word in newFile]'''

            lemmatizer = WordNetLemmatizer()
            newFile=word_tokenize(newFile)
            newFile = [lemmatizer.lemmatize(word) for word in newFile]

            newFile = ' '.join([num2words.num2words(i) if i.isdigit() else i for i in newFile])

            for i in newFile:
                emojis = demoji.findall(i)
                if i in emojis:
                    newFile = newFile.replace(i,emojis[i])
                #print(row)
            newFile = ''.join(i for i in newFile)

            with open (allfile, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([newFile])

for date in daterange(start_date, end_date):
    after = date + timedelta(1)
    #print(date.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv")
    filename="withcases"+date.strftime("%B%-d").lower()+".csv"
    print(filename)
    allfile = "preprocessed_corona_tweets"+date.strftime("%B%-d").lower()+".csv"
    with open(filename, newline='') as csvfiles:
        files = csv.reader(csvfiles, delimiter=' ', quotechar='|')
        totaldata=pd.read_csv(filename, header=None)
        dataframe = totaldata[1]
        for text in dataframe:
            #print(type(text))
            newFile = np.char.lower(str(text))
            #print("lower: ", newFile)

            newFile = remove_stopwords(str(newFile))
            #print("stop: ", newFile)

            newFile = re.sub('http[s]?://\S+', '', str(newFile))
            #print("URL: ", newFile)

            newFile = np.char.replace(newFile, '\\n', '')
            newFile = np.char.replace(newFile, '\\r', '')

            symbols = "!#$%&()*+-"
            for i in symbols:
                newFile = np.char.replace(newFile, i, ' ')
            symbols = "!./:;<=>?@[\]^_`{|}~"
            for i in range(len(symbols)):
                newFile = np.char.replace(newFile, symbols[i], ' ')
            newFile = np.char.replace(newFile, ',', '')
            newFile = np.char.replace(newFile, "  ", " ")
            #print("punc: ", newFile)
            #print (newFile)
            newFile = np.char.replace(str(newFile), "’", "")

            newFile = re.sub(r"\b[a-zA-Z]\b", "", str(newFile))
            '''ps = PorterStemmer()
            newFile = [ps.stem(word) for word in newFile]'''
            lemmatizer = WordNetLemmatizer()
            newFile=word_tokenize(newFile)
            newFile = [lemmatizer.lemmatize(word) for word in newFile]

            newFile = ' '.join([num2words.num2words(i) if i.isdigit() else i for i in newFile])

            for i in newFile:
                emojis = demoji.findall(i)
                if i in emojis:
                    newFile = newFile.replace(i,emojis[i])
                #print(row)
            newFile = ''.join(i for i in newFile)

            with open (allfile, 'a', newline='') as csvfiless:
                csvwriter = csv.writer(csvfiless)
                csvwriter.writerow([newFile])   

    break


for date in daterange(start_date, end_date):
    tweet_filename="preprocessed_corona_tweets"+date.strftime("%B%-d").lower()+".csv"
    cases_filename="withcases"+date.strftime("%B%-d").lower()+".csv"

    df_tweets=pd.read_csv(tweet_filename)
    df_cases=pd.read_csv(cases_filename)
    df_cases=df_cases.drop(columns = ["Province_State","Unnamed: 0"])

    new_df=pd.concat([df_tweets, df_cases],axis = 1)
    new_df.to_csv("finalTweets"+date.strftime("%B%d").lower()+".csv")
    break