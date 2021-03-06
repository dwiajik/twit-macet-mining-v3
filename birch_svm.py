import csv
import gc
from multiprocessing import Pool
from os.path import dirname, join
from random import shuffle

import numpy as np
from sklearn.cluster import Birch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

from modules.tokenizer import ngrams_tokenizer
from modules.cleaner import clean
from modules.similarity import *

print('PROGRESS: Initializing...')

count_vect = CountVectorizer(preprocessor=clean)
clf = LinearSVC(max_iter=10000)
splitBy = 20
calculations = [
    Cosine(),
    Dice(),
    Jaccard(),
    Overlap(),
]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ngrams = 1

print('PROGRESS: Loading datasets...')

with open(join(dirname(__file__), 'result/generated_datasets/traffic.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    traffic_tweets = [line[0] for line in dataset]

with open(join(dirname(__file__), 'result/generated_datasets/non_traffic.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    non_traffic_tweets = [line[0] for line in dataset]

with open(join(dirname(__file__), 'tweets_corpus/test_set_10000.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    dataset = [(line[0], line[1]) for line in dataset]
    shuffle(dataset)
    test = {
        'data': [line[0] for line in dataset],
        'target': [line[1] == 'traffic' for line in dataset],
    }

print('PROGRESS: Shuffling datasets...')

shuffle(traffic_tweets)
shuffle(non_traffic_tweets)

print('PROGRESS: Reducing datasets by 1/{} ...'.format(splitBy))

traffic_tweets_size = int(len(traffic_tweets) / splitBy)
non_traffic_tweets_size = int(len(non_traffic_tweets) / splitBy)

traffic_tweets = traffic_tweets[:traffic_tweets_size]
non_traffic_tweets = non_traffic_tweets[:non_traffic_tweets_size]

print('PROGRESS: Extracting features of datasets...')

vectors = count_vect.fit_transform(traffic_tweets + non_traffic_tweets)
print('\tAll {} data feature vector shape: {}'.format(traffic_tweets_size + non_traffic_tweets_size, vectors.shape))

traffic_vectors = vectors[:traffic_tweets_size]
non_traffic_vectors = vectors[traffic_tweets_size:]
print('\tTraffic data feature vector shape: {}'.format(traffic_vectors.shape))
print('\tNon traffic data feature vector shape: {}'.format(non_traffic_vectors.shape))

test_vectors = count_vect.transform(test['data'])
print('\tTest data feature vector shape: {}'.format(test_vectors.shape))

print('PROGRESS: Train SVM with all the data...')

target = [True] * len(traffic_tweets) + [False] * len(non_traffic_tweets)
clf.fit(vectors, target)

print('PROGRESS: Evaluate the SVM model using test set...')

predicted = clf.predict(test_vectors)
accuracy = np.mean(predicted == test['target'])

prfs = precision_recall_fscore_support(test['target'], predicted)

# print('Training time: {}'.format(training_time))
print('\tAccuracy: {}'.format(accuracy))
print('\tPrecision: {}'.format(prfs[0][0]))
print('\tRecall: {}'.format(prfs[1][0]))
print('\tF-score: {}'.format(prfs[2][0]))
with open(join(dirname(__file__), 'birch_eval.csv'), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    csv_writer.writerow(('all', '', len(traffic_tweets), len(non_traffic_tweets), accuracy, prfs[0][0], prfs[1][0], prfs[2][0]))

for th in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
    brc = Birch(branching_factor=50, n_clusters=None, threshold=th, compute_labels=True)
    print('PROGRESS: Clustering dataset...')

    brc.fit(traffic_vectors)
    traffic_vectors = brc.subcluster_centers_
    print('\tTraffic data centroids count: {}'.format(len(traffic_vectors)))

    brc.fit(non_traffic_vectors)
    non_traffic_vectors = brc.subcluster_centers_
    print('\tNon traffic data centroids count: {}'.format(len(non_traffic_vectors)))

    training_vectors = np.concatenate((traffic_vectors, non_traffic_vectors))
    training_target = [True] * len(traffic_vectors) + [False] * len(non_traffic_vectors)

    print('\tTotal centroid count: {}'.format(len(training_vectors)))

    print('PROGRESS: Train SVM with cluster centroids...')

    clf.fit(training_vectors, training_target)

    print('PROGRESS: Evaluate the SVM model using test set...')

    predicted = clf.predict(test_vectors)
    accuracy = np.mean(predicted == test['target'])

    prfs = precision_recall_fscore_support(test['target'], predicted)

    # print('Training time: {}'.format(training_time))
    print('\tAccuracy: {}'.format(accuracy))
    print('\tPrecision: {}'.format(prfs[0][0]))
    print('\tRecall: {}'.format(prfs[1][0]))
    print('\tF-score: {}'.format(prfs[2][0]))
    with open(join(dirname(__file__), 'birch_eval.csv'), 'a', newline='\n') as csv_output:
        csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        csv_writer.writerow(('birch', th, len(traffic_vectors), len(non_traffic_vectors), accuracy, prfs[0][0], prfs[1][0], prfs[2][0]))

    brc = None
    gc.collect()

def calculate(calculation):
    r = []
    for threshold in thresholds:
        print('PROGRESS: Reduce dataset using {} - {}...'.format(calculation.__class__.__name__, threshold))

        cleaned_traffic_tweets = [(tweet, clean(tweet)) for tweet in traffic_tweets]
        tokenized_traffic_tweets = [(tweet, ngrams_tokenizer(cleaned, ngrams)) for (tweet, cleaned) in cleaned_traffic_tweets]

        cleaned_non_traffic_tweets = [(tweet, clean(tweet)) for (tweet) in non_traffic_tweets]
        tokenized_non_traffic_tweets = [(tweet, ngrams_tokenizer(cleaned, ngrams)) for (tweet, cleaned) in cleaned_non_traffic_tweets]

        distinct_traffic_tweets = []
        distinct_non_traffic_tweets = []

        for (tweet, tokens) in tokenized_traffic_tweets:
            if len(distinct_traffic_tweets) == 0:
                distinct_traffic_tweets.append((tweet, tokens))
            else:
                is_new = True
                for (tweet2, tokens2) in distinct_traffic_tweets:
                    score = calculation.index(tokens, tokens2)
                    if score >= threshold:
                        is_new = False
                if is_new:
                    distinct_traffic_tweets.append((tweet, tokens))

        for (tweet, tokens) in tokenized_non_traffic_tweets:
            if len(distinct_non_traffic_tweets) == 0:
                distinct_non_traffic_tweets.append((tweet, tokens))
            else:
                is_new = True
                for (tweet2, tokens2) in distinct_non_traffic_tweets:
                    score = calculation.index(tokens, tokens2)
                    if score >= threshold:
                        is_new = False
                if is_new:
                    distinct_non_traffic_tweets.append((tweet, tokens))

        distinct_traffic_tweets = [tweet for (tweet, tokens) in distinct_traffic_tweets]
        distinct_non_traffic_tweets = [tweet for (tweet, tokens) in distinct_non_traffic_tweets]

        print('\tTraffic tweets count: {}'.format(len(distinct_traffic_tweets)))
        print('\tNon traffic tweets count: {}'.format(len(distinct_non_traffic_tweets)))

        training_vectors = count_vect.fit_transform(distinct_traffic_tweets + distinct_non_traffic_tweets)
        training_target = [True] * len(distinct_traffic_tweets) + [False] * len(distinct_non_traffic_tweets)

        test_vectors = count_vect.transform(test['data'])

        print('PROGRESS: Train SVM with reduced dataset using {} - {}...'.format(calculation.__class__.__name__, threshold))

        clf.fit(training_vectors, training_target)

        print('PROGRESS: Evaluate the SVM model using test set...')

        predicted = clf.predict(test_vectors)
        accuracy = np.mean(predicted == test['target'])

        prfs = precision_recall_fscore_support(test['target'], predicted)

        # print('Training time: {}'.format(training_time))
        print('\tAccuracy: {}'.format(accuracy))
        print('\tPrecision: {}'.format(prfs[0][0]))
        print('\tRecall: {}'.format(prfs[1][0]))
        print('\tF-score: {}'.format(prfs[2][0]))

        r.append((calculation.__class__.__name__, threshold, len(distinct_traffic_tweets), len(distinct_non_traffic_tweets), accuracy, prfs[0][0], prfs[1][0], prfs[2][0]))

    return r

p = Pool(4)
results = p.map(calculate, calculations)

with open(join(dirname(__file__), 'birch_eval.csv'), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for result in results:
        for r in result:
            csv_writer.writerow(r)
