import csv
import os
import pickle
import random
import sys

from modules import cleaner, tokenizer

def tweet_features(tweet):
    features = {}
    tweet = cleaner.clean(tweet)

    for word in tweet.split():
        features["{}".format(word)] = tweet.count(word)

    return features

if sys.argv[1] == 'original':
    traffic_tweets = [(line, 'traffic') for line in open('tweets_corpus/traffic_tweets_combined.txt')]
    non_traffic_tweets = [(line, 'non_traffic') for line in open('tweets_corpus/random_tweets.txt')] + \
        [(line, 'non_traffic') for line in open('tweets_corpus/non_traffic_tweets.txt')]

elif sys.argv[1] == 'clean':
    with open(os.path.join(os.path.dirname(__file__), 'distinct_traffic_tweets.csv'), newline='\n') as csv_input:
        dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
        traffic_tweets = [(line[0], line[1]) for line in dataset]

    with open(os.path.join(os.path.dirname(__file__), 'distinct_non_traffic_tweets.csv'), newline='\n') as csv_input:
        dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
        non_traffic_tweets = [(line[0], line[1]) for line in dataset]

random.shuffle(traffic_tweets)
random.shuffle(non_traffic_tweets)

labeled_tweets = (traffic_tweets + non_traffic_tweets)
random.shuffle(labeled_tweets)

train_set = [(tweet_features(tweet), category) for (tweet, category) in labeled_tweets]

with open(sys.argv[2], 'wb') as output:
    pickle.dump(train_set, output, pickle.HIGHEST_PROTOCOL)