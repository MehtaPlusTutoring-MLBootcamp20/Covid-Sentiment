from datetime import date, timedelta, datetime
import time
import csv
import pandas as pd
import tweepy

def daterange(start_date, end_date):
    #for n in range(int((end_date - start_date).days)):
        #yield start_date + timedelta(n)
start_date = date(2020, 4, 11)
end_date = date(2020, 7, 12) #end date, datetime.date(datetime.now()) (this second option is dynamic and changes by date but depends on timezone)

for date in daterange(start_date, end_date):
    after = date + timedelta(1)
    print("ready_corona_tweets"+after.strftime("%B%-d").lower()+".csv")
    filename=date.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv"
    if (date.strftime("%B%-d").lower() != "march29"):
        with open(date.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv", 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=' ', quotechar='|')

def extract_place(status):
    if type(status) is tweepy.models.Status:
        status = status.__dict__
    #Try to get the place from the place data inside the status dict
    if status['place'] is not None:
        place = status['place']
        if place['country'] != 'United States':
            return place['country']
        elif place['place_type'] == 'admin':
            return place['name']
        elif place['place_type'] == 'city':
            return filename.get(place['full_name'].split(', ')[-1])
    #If the status dict has no place info, get the place from the user data
    else:
        place = status['user']['location']
        try:
            place = place.split(', ')[-1].upper()
        except AttributeError:
            return None
        if place in filename:
            return filename[place]
        else:
            return place

