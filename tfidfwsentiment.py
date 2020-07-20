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
articles = articles.drop(columns = ['Unnamed: 0','tweet.1'])

#print(articles.head(5))
#print(articles.columns)

#print (articles.head(10))
articlesFile = "articles.csv"
articles.to_csv(articlesFile)
#artic = pd.read_csv(articlesFile)
#artic = artic.drop(columns = ['Unnamed: 0'])
#result = pd.concat([df1, df4], ignore_index=True
#result = pd.concat([sentiment, artic], axis = 1)
#print (artic.head())
#print(result.head())
#result.to_csv("articles.csv", index=False)