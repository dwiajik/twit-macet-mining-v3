import csv
import os

from modules import cleaner, tokenizer
from modules.similarity import *

calculation = Overlap()
ngrams = 1
threshold = 0.8

traffic_tweets = [(line, 'traffic') for line in open('tweets_corpus/traffic_tweets_combined.txt')]
non_traffic_tweets = [(line, 'non_traffic') for line in open('tweets_corpus/random_tweets.txt')] + \
    [(line, 'non_traffic') for line in open('tweets_corpus/non_traffic_tweets.txt')]

print('traffic tweets: {}'.format(len(traffic_tweets)))
print('non_traffic tweets: {}'.format(len(non_traffic_tweets)))

# labeled_tweets = traffic_tweets + non_traffic_tweets

cleaned_traffic_tweets = [(tweet, category, cleaner.clean(tweet)) for (tweet, category) in traffic_tweets]
tokenized_traffic_tweets = [(tweet, category, tokenizer.ngrams_tokenizer(cleaned, ngrams)) for (tweet, category, cleaned) in cleaned_traffic_tweets]

progress = 0
distinct_traffic_tweets = []
for (tweet, category, tokens) in tokenized_traffic_tweets:
    progress += 1
    print('\r{}/{}'.format(progress, len(tokenized_traffic_tweets)), end='')

    if len(distinct_traffic_tweets) == 0:
        distinct_traffic_tweets.append((tweet, category, tokens))
    else:
        is_new = True
        for (tweet2, category2, tokens2) in distinct_traffic_tweets:
            score = calculation.index(tokens, tokens2)
            if score >= threshold:
                is_new = False

        if is_new:
            distinct_traffic_tweets.append((tweet, category, tokens))

cleaned_non_traffic_tweets = [(tweet, category, cleaner.clean(tweet)) for (tweet, category) in non_traffic_tweets]
tokenized_non_traffic_tweets = [(tweet, category, tokenizer.ngrams_tokenizer(cleaned, ngrams)) for (tweet, category, cleaned) in cleaned_non_traffic_tweets]

progress = 0
distinct_non_traffic_tweets = []
for (tweet, category, tokens) in tokenized_non_traffic_tweets:
    progress += 1
    print('\r{}/{}'.format(progress, len(tokenized_non_traffic_tweets)), end='')

    if len(distinct_non_traffic_tweets) == 0:
        distinct_non_traffic_tweets.append((tweet, category, tokens))
    else:
        is_new = True
        for (tweet2, category2, tokens2) in distinct_non_traffic_tweets:
            score = calculation.index(tokens, tokens2)
            if score >= threshold:
                is_new = False

        if is_new:
            distinct_non_traffic_tweets.append((tweet, category, tokens))

print('Result:')
print('traffic tweets: {}'.format(len(distinct_traffic_tweets)))
print('non_traffic tweets: {}'.format(len(distinct_non_traffic_tweets)))

with open(os.path.join(os.path.dirname(__file__), 'distinct_traffic_tweets.csv'), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for tweet in distinct_traffic_tweets:
        csv_writer.writerow(tweet)

with open(os.path.join(os.path.dirname(__file__), 'distinct_non_traffic_tweets.csv'), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for tweet in distinct_non_traffic_tweets:
        csv_writer.writerow(tweet)
