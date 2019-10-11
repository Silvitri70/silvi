#CROWLING TWITTER

import tweepy           # T
import pandas as pd     
import numpy as np      

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import datetime
import translator
import json
import googletrans

import re

from googletrans import Translator
Translator = Translator()

ACCESS_TOKEN = "1151666277036859392-SWXz8IOVZ9ijCb3TVodakiO7xx5Xho"
ACCESS_TOKEN_SECRET = "8nJrshCFrGcwHCATBKuqciCPHcPbgR9UYs0gsaUfLH79h"
CONSUMER_KEY = "GrUzSnJ90yxhHqgRqTKdQH7J4"
CONSUMER_SECRET = "ZczgmhYz4iHV8W35MC1nf2LNboEUB8fr7xCps3L6dAd9spPdxc"

query=["FadliZon"] # Enter user or query

def twitter_setup():
   
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

   
    api = tweepy.API(auth)
    return api
extractor = twitter_setup()
counter = 0

for query_each in query:    
    

    public_tweets = extractor.search(q=query_each, count=100, geocode="-6.2293867,106.6894293,100km",parser=tweepy.parsers.JSONParser())
    
    if counter<1:
        tweets = public_tweets['statuses']
        counter = counter+1
    else:
        tweets.extend(public_tweets['statuses'])

         
    print(''+query_each)
    print('Number of tweets extracted: {}.\n'.format(len(public_tweets['statuses'])))

    print("5 recent tweets:\n")
    for tweet in public_tweets['statuses'][:5]:
        print(tweet['text'])
        print()
        
with open('fa.json', 'w') as json_file:
    json.dump(tweets, json_file)
    
# We create a pandas dataframe as follows:
data = pd.DataFrame(data=[tweet['text'] for tweet in tweets], columns=['Tweets'])

# We display the first 10 elements of the dataframe:
display(data.head(10))

# Internal methods of a single tweet object:
print(dir(tweets[0]))

data['len']  = np.array([len(tweet['text']) for tweet in tweets])
data['ID']   = np.array([tweet['id'] for tweet in tweets])
data['Date'] = np.array([tweet['created_at'] for tweet in tweets])
data['Source'] = np.array([tweet['source'] for tweet in tweets])
data['Likes']  = np.array([tweet['favorite_count'] for tweet in tweets])
data['RTs']    = np.array([tweet['retweet_count'] for tweet in tweets])
data['Language'] = np.array([tweet['lang'] for tweet in tweets])

counter_source = 0;
for source in data['Source']:

    clean = re.compile('<.*?>')   
    data['Source'][counter_source] = re.sub(clean, '', source)
    counter_source = counter_source+1

# Display of first 10 elements from dataframe:
display(data.head(10))

mean = np.mean(data['len'])

print("The lenght's average in tweets: {}".format(mean))

tweets_by_lang = data['Language'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Languages', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 languages', fontsize=15, fontweight='bold')
tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')

fav_max = np.max(data['Likes'])
rt_max  = np.max(data['RTs'])

fav = data[data.Likes == fav_max].index[0]
rt  = data[data.RTs == rt_max].index[0]

# Max FAVs:
print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
print("Number of likes: {}".format(fav_max))
print("{} characters.\n".format(data['len'][fav]))

# Max RTs:
print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
print("Number of retweets: {}".format(rt_max))
print("{} characters.\n".format(data['len'][rt]))

tlen = pd.Series(data=data['len'].values, index=data['Date'])
tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
tret = pd.Series(data=data['RTs'].values, index=data['Date'])

tlen.plot(figsize=(16,4), color='r');

tfav.plot(figsize=(16,4), label="Likes", legend=True)
tret.plot(figsize=(16,4), label="Retweets", legend=True);

sources = []
for source in data['Source']:
    
    if source not in sources:        
        sources.append(source)


print("Creation of content sources:")
for source in sources:    
    print(source)

percent = np.zeros(len(sources))

for source in data['Source']:
    
    for index in range(len(sources)):
        if source == sources[index]:
            percent[index] += 1
            pass

percent /= 10


pie_chart = pd.Series(percent, index=sources, name='Sources')
pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(10,8));

from textblob import TextBlob
import re
# Import google translate
from googletrans import Translator
translator = Translator()

def clean_tweet(tweet):
       
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

#translate tweet 
def translate_bahasa(tweet):
    tweet = translator.translate(tweet.encode('utf-8').decode('ascii',errors='ignore'), src='id', dest='en') 
    tweet = tweet.text
    return tweet

def analize_sentiment(tweet):
    
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

!pip install textblob

data['Translate'] = np.array([translate_bahasa(tweet) for tweet in data['Tweets']])


data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Translate'] ])
data['SAID'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])

display(data.head(10))

pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]

pos_tweets_id = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SAID'][index] > 0]
neu_tweets_id = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SAID'][index] == 0]
neg_tweets_id = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SAID'][index] < 0]

print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
print("Percentage of negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))

print('Sentimen Analisis Sesudah diTranslate ke bahasa inggris')
# Data 
labels = 'Positive Tweets', 'Neutral Tweets','Negative Tweets'
sizes = [len(pos_tweets)*100/len(data['Tweets']), len(neu_tweets)*100/len(data['Tweets']), len(neg_tweets)*100/len(data['Tweets'])]
colors = ['cyan', 'red', 'lightcoral']
explode = (0.07, 0.07, 0.07)  # explode 1st slice
 
# Plot
plt.figure(figsize=(10,6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

#BAHASA INDO
print('Sentimen Analisis Sebelum diTranslate')
# Data to plot
labels = 'Positive Tweets', 'Neutral Tweets','Negative Tweets'
sizes = [len(pos_tweets_id)*100/len(data['Tweets']), len(neu_tweets_id)*100/len(data['Tweets']), len(neg_tweets_id)*100/len(data['Tweets'])]
colors = ['gold', 'magenta', 'lightcoral']
explode = (0.07, 0.07, 0.07)  # explode 1st slice
 
# Plot
plt.figure(figsize=(10,6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

print('Sentimen Analisis Sebelum diTranslate')
# Data to plot
labels = 'Positive Tweets', 'Neutral Tweets','Negative Tweets'
sizes = [len(pos_tweets_id)*100/len(data['Tweets']), len(neu_tweets_id)*100/len(data['Tweets']), len(neg_tweets_id)*100/len(data['Tweets'])]
colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0.07, 0.07, 0.07)  # explode 1st slice
