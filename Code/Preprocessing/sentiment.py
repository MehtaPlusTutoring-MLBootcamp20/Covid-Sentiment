import numpy as np 
import pandas as pd 
from datetime import date, timedelta, datetime
import time
import csv

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
start_date = date(2020, 4, 12)
end_date = date(2020, 7, 13) 

sentiments = pd.DataFrame()
for date in daterange(start_date, end_date):
    after = date + timedelta(1)
    print(date.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv")
    filename=date.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv"
    with open(filename, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        totaldata = pd.read_csv(filename, header=None)
        dataframe = totaldata[1]
        sentiments = pd.concat([sentiments, dataframe],ignore_index=True)
'''
for date in daterange(start_date, end_date):
    after = date + timedelta(1)
    print(date.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv")
    filename=date.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv"
    if (date.strftime("%B%-d").lower() != "march29"):
        with open(date.strftime("%B%-d").lower()+"_"+after.strftime("%B%-d").lower()+".csv", 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=' ', quotechar='|')
            totaldata=pd.read_csv(filename, header=None)
            dataframe=totaldata[0]
'''
sentimentFile = "sentiment.csv"
sentiments.to_csv(sentimentFile, index = False)