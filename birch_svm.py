import csv
from os.path import dirname, join
from random import shuffle

import numpy as np
from sklearn.cluster import Birch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

from modules.cleaner import clean

count_vect = CountVectorizer(preprocessor=clean)
clf = LinearSVC(max_iter=10000)

with open(join(dirname(__file__), 'result/generated_datasets/traffic.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    traffic_tweets = [line[0] for line in dataset]

with open(join(dirname(__file__), 'result/generated_datasets/non_traffic.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    non_traffic_tweets = [line[0] for line in dataset]

shuffle(traffic_tweets)
shuffle(non_traffic_tweets)

traffic_tweets_size = int(len(traffic_tweets) / 20)

traffic_tweets = traffic_tweets[:traffic_tweets_size]
non_traffic_tweets = non_traffic_tweets[:int(len(non_traffic_tweets) / 20)]

vectors = count_vect.fit_transform(traffic_tweets + non_traffic_tweets)
print(vectors.shape)

traffic_vectors = vectors[:traffic_tweets_size]
non_traffic_vectors = vectors[traffic_tweets_size:]
print(traffic_vectors.shape)
print(non_traffic_vectors.shape)

brc = Birch(branching_factor=50, n_clusters=None, threshold=2.5, compute_labels=True)

brc.fit(traffic_vectors)
traffic_vectors = brc.subcluster_centers_
print(len(traffic_vectors))

brc.fit(non_traffic_vectors)
non_traffic_vectors = brc.subcluster_centers_
print(len(non_traffic_vectors))

training_vectors = np.concatenate((traffic_vectors, non_traffic_vectors))
training_target = [True] * len(traffic_vectors) + [False] * len(non_traffic_vectors)

print(len(training_vectors))

clf.fit(training_vectors, training_target)

## Open Test set
with open(join(dirname(__file__), 'tweets_corpus/test_set_10000.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    dataset = [(line[0], line[1]) for line in dataset]
    shuffle(dataset)
    test = {
        'data': [line[0] for line in dataset],
        'target': [line[1] == 'traffic' for line in dataset],
    }

test_vectors = count_vect.transform(test['data'])

predicted = clf.predict(test_vectors)
accuracy = np.mean(predicted == test['target'])

prfs = precision_recall_fscore_support(test['target'], predicted)

# print('Training time: {}'.format(training_time))
print('Accuracy: {}'.format(accuracy))
print('Precision: {}'.format(prfs[0][0]))
print('Recall: {}'.format(prfs[1][0]))
print('F-score: {}'.format(prfs[2][0]))