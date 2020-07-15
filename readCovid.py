from datetime import date, timedelta, datetime
import time
import csv
import pandas as pd
from twarc import Twarc
#from geopy.geocoders import Nominatim
#import countries

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

#cc = countries.CountryChecker('TM_WORLD_BORDERS-0.3.shp')

for date in daterange(start_date, end_date):
    after = date + timedelta(1)
    print(date.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv")
    filename=date.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv"
    if (date.strftime("%B%-d").lower() != "march29"):
        with open(date.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv", 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=' ', quotechar='|')
            """
            for row in data:
                print (row)
                break    
            """

            
            totaldata=pd.read_csv(filename, header=None)
            dataframe=totaldata[0]
            numberfile = "number_corona_tweets"+ date.strftime("%B%-d").lower() +".txt"
            readyfile = "ready_corona_tweets"+ date.strftime("%B%-d").lower() +".csv"
            dataframe.to_csv(numberfile, index=False, header=None)
            for tweet in t.hydrate(open(numberfile)):
                #print (tweet["place"]["country"])
                if (tweet["place"] == None):
                    continue
                if (tweet["place"]["country"] == None):
                    continue
                '''
                if ("place" not in tweet):
                    print ("gibberish")
                    continue
                if ("country" not in tweet["place"]):
                    #print ("sdjfksldf")
                    continue
                '''
                if (tweet["place"]["country"] == "United States"):
                    #print("sdkjf")
                    with open (readyfile, 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=' ', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
                        csvwriter.writerow([tweet["full_text"]])
                    #wr = open(readyfile, 'a')
                    #wr.write(tweet["full_text"])
                #print(tweet["full_text"])
                #break
            
    break

'''
open()
with open('eggs.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
'''
