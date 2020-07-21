from wordcloud import WordCloud,STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt 
stopwords = set(STOPWORDS)

from textblob import TextBlob

meow = pd.read_csv("articles2.csv")

def show_wordcloud(data , title = None):
    wordcloud = WordCloud(background_color='white',stopwords=stopwords,max_words=200,max_font_size=40).generate(str(data))
  
    #fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    plt.title(title, size = 25)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

show_wordcloud(meow['tweet'])