from datetime import date, timedelta, datetime
import time
import csv
import pandas as pd
import plotly.express as px

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
start_date = date(2020, 4, 11)
end_date = date(2020, 7, 13)

with open('averageCase.csv', 'w+') as csvfile:
    csvwriter = csv.writer(csvfile)
    header = ['Date','Average Cases']
    csvwriter.writerow(i for i in header)
    for dates in daterange(start_date, end_date):
        after = dates + timedelta(1)
        filename="new"+after.strftime("%m") + "-" + after.strftime("%d") + "-2020.csv"
        filed= pd.read_csv(filename)
        files = filed.drop(filed.columns[0],axis=1)
        casesum = 0
        for i in range(51):
            casesum += files.iat[i,1]
        average = casesum/51
        csvwriter.writerow([after,str(average)])

df = pd.read_csv('averageCase.csv')    
graph = px.line(df, x = 'Date', y = 'Average Cases', title = 'Average COVID-19 Cases vs. Time')
graph.show()
    