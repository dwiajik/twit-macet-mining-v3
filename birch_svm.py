import csv
from os.path import dirname, join
from random import shuffle

import numpy as np
from sklearn.cluster import Birch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

from modules.cleaner import clean

print('PROGRESS: Initializing...')

count_vect = CountVectorizer(preprocessor=clean)
clf = LinearSVC(max_iter=10000)
brc = Birch(branching_factor=50, n_clusters=None, threshold=2.5, compute_labels=True)
splitBy = 20

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