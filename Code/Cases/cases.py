from datetime import date, timedelta, datetime
import time
import csv
import pandas as pd
#from geopy.geocoders import Nominatim
#import countries

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
start_date = date(2020, 4, 11)
end_date = date(2020, 7, 13) #end date, datetime.date(datetime.now()) (this second option is dynamic and changes by date but depends on timezone)


#cc = countries.CountryChecker('TM_WORLD_BORDERS-0.3.shp')

for dates in daterange(start_date, end_date):
    after = dates + timedelta(1)
    #print (after)
    print(after.strftime("%m") + "-" + after.strftime("%d") + "-2020.csv")
    filename=after.strftime("%m") + "-" + after.strftime("%d") + "-2020.csv"
    filed= pd.read_csv(filename)
    files = filed.drop(columns = ["Country_Region","Last_Update","Lat","Long_","Deaths","Recovered","Active","FIPS","Incident_Rate","People_Tested","People_Hospitalized","Mortality_Rate","UID","ISO3","Testing_Rate","Hospitalization_Rate"])
    #print(files.columns)
    #files = files.drop([8,12,13,42,55,56,57,58])
    lists=["Puerto Rico","Diamond Princess","Grand Princess","Guam","American Samoa","Northern Mariana Islands","Recovered","Virgin Islands"]
    #print (lists)
    indexNames = files[files["Province_State"].isin(lists)].index
    files.drop(indexNames, inplace=True)
    #print(files.head(10))
    files.to_csv("new" + filename)
    #print ("skdfjdslkf")
    #print(pd.set_option('dsplay.max_rows', 10))
    #with open(filename, 'a') as csvfile:
    #        data = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #break

'''
open()
with open('eggs.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
'''
