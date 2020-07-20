from datetime import date, timedelta, datetime
import time
import csv
import pandas as pd
from twarc import Twarc

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
start_date = date(2020, 4, 12)
end_date = date(2020, 7, 13) #end date, datetime.date(datetime.now()) (this second option is dynamic and changes by date but depends on timezone)

OAUTH_TOKEN = "1029186921438883845-AQjxqWPxZlURJ47eWFqRFRkSCkDPFh"
OAUTH_TOKEN_SECRET = "YgxeTz31ItxBrJubvwZpZaqa57LLhWRKLMM4t82pdEtsv"
CONSUMER_KEY = "Y70ckEEL2TdQzyq9NqI5RriiB"
CONSUMER_SECRET = "YWQJJlJyzXxkaPXCEdFrANgHFf4Dyd0PtkT4f5TvXFUJLUtpvU"
t = Twarc(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

for singledate in daterange(start_date, end_date):
    after = singledate + timedelta(1)
    filename=singledate.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv"
    if (singledate.strftime("%B%-d").lower() != "march29"):
        with open(filename, 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=' ', quotechar='|')
            totaldata=pd.read_csv(filename, header=None)
            dataframe=totaldata[0]
            sentimentstuff = totaldata[1]
            numberfile = "number_corona_tweets_state"+ singledate.strftime("%B%-d").lower() +".txt"
            readyfile = "sentiment"+ singledate.strftime("%B%-d").lower() +".csv"
            dataframe.to_csv(numberfile, index=False, header=None)

            with open (readyfile, 'w') as csvfile:
                fieldnames = ['tweet', 'sentiment']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            counter = 0
            for tweet in t.hydrate(open(numberfile)):
                if (tweet["place"] == None):
                    continue
                if (tweet["place"]["country"] == "United States"):   

                    # with open (readyfile, 'a') as csvfile:
                    #     fieldnames = ['tweet', 'location']
                    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    #     writer.writeheader()
                    with open(readyfile,'a') as csvfile:
                        try:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow({'tweet': tweet["full_text"], 'sentiment': sentimentstuff[counter]})
                        except:
                            pass
                counter = counter + 1