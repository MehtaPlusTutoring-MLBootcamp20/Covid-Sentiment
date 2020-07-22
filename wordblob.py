from wordcloud import WordCloud,STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt 
stopwords = set(STOPWORDS)

from textblob import TextBlob

meow = pd.read_csv("articles2.csv")

meow.loc[(meow["sentiment"] < -1/10) & (meow["sentiment"] >= -1) , "sentiment"] = -1
meow.loc[(meow["sentiment"] < 1/10) & (meow["sentiment"] >= -1/10) , "sentiment"] = 0
meow.loc[(meow["sentiment"] <= 1) & (meow["sentiment"] >= 1/10) , "sentiment"] = 1

positive_df = meow[meow['sentiment'] == 1]
neutral_df = meow[meow['sentiment'] == 0]
neg_df = meow[meow['sentiment'] == -1]

def show_wordcloud(data , title = None):
    #wordcloud = WordCloud(background_color='skyblue',stopwords=stopwords,max_words=200,max_font_size=40).generate(str(data))
    wordcloud = WordCloud(background_color='white',colormap="Greens",max_words=200,max_font_size=40).generate(str(data))

    #fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    plt.title(title, size = 25)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

show_wordcloud(meow['tweet'])
#show_wordcloud(positive_df)
#show_wordcloud(neutral_df)
#show_wordcloud(neg_df)