import csv
from os.path import dirname, join
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC

from modules.cleaner import clean
categories = ['traffic', 'non_traffic']
count_vect = CountVectorizer(preprocessor=clean)

with open(join(dirname(__file__), 'result/generated_datasets/traffic.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    traffic_tweets = [line[0] for line in dataset]

with open(join(dirname(__file__), 'result/generated_datasets/non_traffic.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    non_traffic_tweets = [line[0] for line in dataset]

tweets = {
    'data': traffic_tweets + non_traffic_tweets,
    'target': [True] * len(traffic_tweets) + [False] * len(non_traffic_tweets),
}

training_vectors = count_vect.fit_transform(tweets['data'])

clf = LinearSVC(max_iter=10000)

cv = StratifiedKFold(n_splits=10, shuffle=True)

scores = cross_val_score(clf, training_vectors, tweets['target'], cv=cv)
print(scores)
print("Accuracy: {} (+/- {})".format(scores.mean(), scores.std() * 2))

scores = cross_val_score(clf, training_vectors, tweets['target'], cv=cv, scoring='precision')
print(scores)
print("Precision: {} (+/- {})".format(scores.mean(), scores.std() * 2))

scores = cross_val_score(clf, training_vectors, tweets['target'], cv=cv, scoring='recall')
print(scores)
print("Recall: {} (+/- {})".format(scores.mean(), scores.std() * 2))

scores = cross_val_score(clf, training_vectors, tweets['target'], cv=cv, scoring='f1')
print(scores)
print("F1: {} (+/- {})".format(scores.mean(), scores.std() * 2))