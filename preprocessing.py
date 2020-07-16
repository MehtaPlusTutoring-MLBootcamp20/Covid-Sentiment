'''
lowercase x
stop words x
punctuation x
apostrophe x
single characters x
stemming x
lemmatisation x
converting numbers x
emojis/emoticons x
'''
#possible big problem!!! changing into line cuts the tweet off!!!!
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
def stopWords (textData):
    noStopWords = 'tempStop.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for text in files:
            newFile = remove_stopwords(str(text))
            #print (newFile)
            with open (noStopWords, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow([newFile])
def urlLink (textData):
    noUrls = 'tempUrl.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for text in files:
            newFile = re.sub('http[s]?://\S+', '', str(text))
            #print (newFile)
            with open (noUrls, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow([newFile])

def punctuation (textData):
    noPunc = 'tempPunc.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for text in files:
            '''newFile = re.sub(rf"[{string.punctuation}]", " ", str(text))
            newFile = np.char.replace(newFile, ',', '')
            #newFile = np.char.replace(newFile, '\n', ' ')'''
            symbols = "!\"#$%&()*+-\n"
            for i in range(len(symbols)):
                newFile = np.char.replace(text, symbols[i], ' ')
                newFile = np.char.replace(newFile, "  ", " ")
            symbols = "!./:;<=>?@[\]^_`{|}~\r"
            for i in range(len(symbols)):
                newFile = np.char.replace(newFile, symbols[i], ' ')
                newFile = np.char.replace(newFile, "  ", " ")
            newFile = np.char.replace(newFile, ',', '')
            with open (noPunc, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow([newFile])
def apos (textData):
    noApos = 'tempApos.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for text in files:
            newFile = np.char.replace(str(text), "’", "")
            #print (newFile)
            with open (noApos, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow([newFile])
def singChar (textData):
    noSing = 'tempSing.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for text in files:
            newFile = re.sub(r"\b[a-zA-Z]\b", "", str(text))
            #print (newFile)
            with open (noSing, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow([newFile])
def stemming(textData):
    noStem = 'tempStem.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for text in files:
            ps = PorterStemmer()
            newFile = ""
            for row in text:
                row=word_tokenize(row)
                newFile = [ps.stem(word) for word in row]
                with open(noStem,'a',newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)  
                    csvwriter.writerow(newFile) 
def lemmatisation(textData):
    noLemma = 'tempLemma.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for text in files:
            lemmatizer = WordNetLemmatizer()
            newFile = ""
            for row in text:
                row=word_tokenize(row)
                newFile = [lemmatizer.lemmatize(word) for word in row]
                with open(noLemma,'a',newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow(newFile)
def numToWord(textData): #only works on pure numbers --> no commas or letters attached (remove the rest?)
    noNum = 'tempNum.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for text in files:
            newFile = ""
            for row in text:
                row = ' '.join([num2words.num2words(i) if i.isdigit() else i for i in row.split()])
                with open(noNum,'a',newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([row])
def emoji(textData): #for some reason us flag not being converted
    noEmo = 'tempEmo.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile,delimiter=' ', quotechar='|')
        for text in files:
            for row in text:
                #print(row)
                for i in row:
                    emojis = demoji.findall(i)
                    if i in emojis:
                        row = row.replace(i,emojis[i])
                #print(row)
                row = ' '.join(i for i in row.split())
                with open(noEmo,'a',newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([row])
#-----------------------------------------------------------------------
def lowerStopUrlPunc (textData):
    allfile = 'tempLowerStopUrlPunc.csv'
    with open(textData, newline='') as csvfile:
        files = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for text in files:
            newFile = np.char.lower(text)
            #print("lower: ", newFile)

            newFile = remove_stopwords(str(newFile))
            #print("stop: ", newFile)

            newFile = re.sub('http[s]?://\S+', '', str(newFile))
            #print("URL: ", newFile)

            symbols = "!\"#$%&()*+-\n"
            for i in range(len(symbols)):
                newFile = np.char.replace(newFile, symbols[i], ' ')
                newFile = np.char.replace(newFile, "  ", " ")
            symbols = "!./:;<=>?@[\]^_`{|}~\r"
            for i in range(len(symbols)):
                newFile = np.char.replace(newFile, symbols[i], ' ')
                newFile = np.char.replace(newFile, "  ", " ")
            newFile = np.char.replace(newFile, ',', '')
            #print("punc: ", newFile)
            #print (newFile)
            newFile = np.char.replace(str(newFile), "’", "")
            with open (allfile, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow([newFile])

lowercase(data)
stopWords(data)
urlLink(data)
punctuation (data)
apos(data)
singChar(data)
lowerStopUrlPunc(data)
stemming(data)
lemmatisation(data)
numToWord(data)
emoji(data)



        #break
#files= pd.read_csv(filename)
#newFile= np.char.lower(files)
#print(newFile)