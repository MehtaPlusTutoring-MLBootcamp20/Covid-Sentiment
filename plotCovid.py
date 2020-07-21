from datetime import date, timedelta, datetime
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
start_date = date(2020, 4, 11)
end_date = date(2020, 7, 13)

with open('averageCase.csv', 'w+') as csvfile:
    csvwriter = csv.writer(csvfile)
    for dates in daterange(start_date, end_date):
        after = dates + timedelta(1)
        filename="new"+after.strftime("%m") + "-" + after.strftime("%d") + "-2020.csv"
        filed= pd.read_csv(filename)
        files = filed.drop(filed.columns[0],axis=1)
        casesum = 0
        for i in range(51):
            casesum += files.iat[i,1]
        average = casesum//51
        csvwriter.writerow([after,str(average)])

with open('newYorkCase.csv','w+') as csvfile:
    csvwriter = csv.writer(csvfile)
    for dates in daterange(start_date,end_date):
        after = dates + timedelta(1)
        filename="new"+after.strftime("%m") + "-" + after.strftime("%d") + "-2020.csv"
        filed = pd.read_csv(filename)
        files = filed.drop(filed.columns[0],axis=1)
        newYork = files.iat[32,1]//1
        csvwriter.writerow([after,str(newYork)])

with open('caliCase.csv','w+') as csvfile:
    csvwriter = csv.writer(csvfile)
    for dates in daterange(start_date,end_date):
        after = dates + timedelta(1)
        filename="new"+after.strftime("%m") + "-" + after.strftime("%d") + "-2020.csv"
        filed = pd.read_csv(filename)
        files = filed.drop(filed.columns[0],axis=1)
        cali = files.iat[4,1]//1
        csvwriter.writerow([after,str(cali)])

with open('floridaCase.csv','w+') as csvfile:
    csvwriter = csv.writer(csvfile)
    for dates in daterange(start_date,end_date):
        after = dates + timedelta(1)
        filename="new"+after.strftime("%m") + "-" + after.strftime("%d") + "-2020.csv"
        filed = pd.read_csv(filename)
        files = filed.drop(filed.columns[0],axis=1)
        florida = files.iat[9,1]//1
        csvwriter.writerow([after,str(florida)])
    

#df = pd.read_csv('averageCase.csv')    
#graph = px.line(df, x = 'Date', y = 'Average Cases', title = 'Average COVID-19 Cases vs. Time')
#graph.show()
avgx = []
avgy = []

newyorkx = []
newyorky = []

calix = []
caliy = []

floridax = []
floriday = []

with open('averageCase.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        avgx.append(row[0])
        avgy.append(int(row[1]))

with open('newYorkCase.csv','r') as csvfile:
    plots = csv.reader(csvfile,delimiter=',')
    for row in plots:
        newyorkx.append(row[0])
        newyorky.append(int(row[1]))

with open('caliCase.csv','r') as csvfile:
    plots = csv.reader(csvfile,delimiter=',')
    for row in plots:
        calix.append(row[0])
        caliy.append(int(row[1]))

with open('floridaCase.csv','r') as csvfile:
    plots = csv.reader(csvfile,delimiter=',')
    for row in plots:
        floridax.append(row[0])
        floriday.append(int(row[1]))

plt.plot(avgx,avgy)
plt.xlabel('Date')
plt.ylabel('Average Cases')
plt.title('Average COVID-19 Cases vs. Time')
plt.yscale('linear')
plt.yticks(np.arange(0, max(avgy), step=10000))
plt.xticks([19,50,80],['May','June','July'])
plt.show()

plt.plot(newyorkx,newyorky)
plt.xlabel('Date')
plt.ylabel('New York Cases')
plt.title('COVID-19 Cases in New York vs. Time')
plt.yscale('linear')
plt.yticks(np.arange(0, max(newyorky), step=100000))
plt.xticks([19,50,80],['May','June','July'])
plt.show()

plt.plot(calix,caliy)
plt.xlabel('Date')
plt.ylabel('California Cases')
plt.title('COVID-19 Cases in California vs. Time')
plt.yscale('linear')
plt.yticks(np.arange(0, max(caliy), step=100000))
plt.xticks([19,50,80],['May','June','July'])
plt.show()

plt.plot(floridax,floriday)
plt.xlabel('Date')
plt.ylabel('Florida Cases')
plt.title('COVID-19 Cases in Florida vs. Time')
plt.yscale('linear')
plt.yticks(np.arange(0, max(floriday), step=100000))
plt.xticks([19,50,80],['May','June','July'])
plt.show()