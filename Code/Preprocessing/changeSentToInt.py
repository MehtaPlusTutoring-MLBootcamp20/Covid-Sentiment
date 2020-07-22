import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

meow = pd.read_csv("articles2.csv")

sentiments = meow["sentiment"]
#sentiments = sentiments.to_frame().T
#print(type(sentiments))

meow.loc[(sentiments < -1/3) & (sentiments >= -1) , "sentiment"] = -1
meow.loc[(sentiments < 1/3) & (sentiments >= -1/3) , "sentiment"] = 0
meow.loc[(sentiments <= 1) & (sentiments >= 1/3) , "sentiment"] = 1

#sentiments.rename(columns={"sentiment": "intSentiment"})

#meow=pd.concat([sentiments, meow], axis = 1)
#print(meow.columns)
meow = meow.drop(columns = ['Unnamed: 0'])

meow.to_csv("articles3.csv")