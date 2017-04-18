'''
We define calculation, ngrams, and threshold on the top. Then the script will strip out the
similar tweets. The output will be list of tweets (containing (tweet, category, tokens)) without retweet.
'''

import csv
import os

from modules import cleaner, tokenizer
from difflib import SequenceMatcher

sm = SequenceMatcher(lambda x: x == " ")

# calculation = Jaccard()
ngrams = 1
threshold = 0.5

traffic_tweets = [(line, 'traffic') for line in open('tweets_corpus/raw/traffic_tweets_combined.txt')]
non_traffic_tweets = [(line, 'non_traffic') for line in open('tweets_corpus/raw/random_tweets.txt')] + \
    [(line, 'non_traffic') for line in open('tweets_corpus/raw/non_traffic_tweets.txt')]

print('traffic tweets: {}'.format(len(traffic_tweets)))
print('non_traffic tweets: {}'.format(len(non_traffic_tweets)))

# labeled_tweets = traffic_tweets + non_traffic_tweets

cleaned_traffic_tweets = [(tweet, category, cleaner.clean(tweet)) for (tweet, category) in traffic_tweets]
# tokenized_traffic_tweets = [(tweet, category, tokenizer.ngrams_tokenizer(cleaned, ngrams)) for (tweet, category, cleaned) in cleaned_traffic_tweets]

progress = 0
distinct_traffic_tweets = []
for (tweet, category, cleaned) in cleaned_traffic_tweets:
    progress += 1
    print('\r{}/{}'.format(progress, len(cleaned_traffic_tweets)), end='')

    if len(distinct_traffic_tweets) == 0:
        distinct_traffic_tweets.append((tweet, category, cleaned))
    else:
        is_new = True
        for (tweet2, category2, cleaned2) in distinct_traffic_tweets:
            # score = calculation.index(tokens, tokens2)
            sm.set_seqs(cleaned, cleaned2)
            score = sm.ratio()
            if score >= threshold:
                is_new = False

        if is_new:
            distinct_traffic_tweets.append((tweet, category, cleaned))

print()

cleaned_non_traffic_tweets = [(tweet, category, cleaner.clean(tweet)) for (tweet, category) in non_traffic_tweets]
# tokenized_non_traffic_tweets = [(tweet, category, tokenizer.ngrams_tokenizer(cleaned, ngrams)) for (tweet, category, cleaned) in cleaned_non_traffic_tweets]

progress = 0
distinct_non_traffic_tweets = []
for (tweet, category, cleaned) in cleaned_non_traffic_tweets:
    progress += 1
    print('\r{}/{}'.format(progress, len(cleaned_non_traffic_tweets)), end='')

    if len(distinct_non_traffic_tweets) == 0:
        distinct_non_traffic_tweets.append((tweet, category, cleaned))
    else:
        is_new = True
        for (tweet2, category2, cleaned2) in distinct_non_traffic_tweets:
            # score = calculation.index(tokens, tokens2)
            sm.set_seqs(cleaned, cleaned2)
            score = sm.ratio()
            if score >= threshold:
                is_new = False

        if is_new:
            distinct_non_traffic_tweets.append((tweet, category, cleaned))

print('Result:')
print('traffic tweets: {}'.format(len(distinct_traffic_tweets)))
print('non_traffic tweets: {}'.format(len(distinct_non_traffic_tweets)))

with open(os.path.join(os.path.dirname(__file__), 'traffic-lcs-0.5.csv'), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for tweet in distinct_traffic_tweets:
        csv_writer.writerow(tweet)

with open(os.path.join(os.path.dirname(__file__), 'non_traffic-lcs-0.5.csv'), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for tweet in distinct_non_traffic_tweets:
        csv_writer.writerow(tweet)
