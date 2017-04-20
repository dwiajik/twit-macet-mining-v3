import csv
from os.path import dirname, join, exists
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC

from modules.cleaner import clean
categories = ['traffic', 'non_traffic']
count_vect = CountVectorizer(preprocessor=clean)
clf = LinearSVC(max_iter=10000)
cv = StratifiedKFold(n_splits=10, shuffle=True)

calculations = [
    'cosine',
    'dice',
    'jaccard',
    'overlap',
    'lcs',
]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

results = []

for calculation in calculations:
    for threshold in thresholds:
        print('{} - {}'.format(calculation, threshold))

        if (exists('result/generated_datasets/{}/{}/traffic.csv'.format(calculation, threshold))):
            with open(join(dirname(__file__), 'result/generated_datasets/{}/{}/traffic.csv'.format(calculation, threshold)), newline='\n') as csv_input:
                dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
                traffic_tweets = [line[0] for line in dataset]

            with open(join(dirname(__file__), 'result/generated_datasets/{}/{}/non_traffic.csv'.format(calculation, threshold)), newline='\n') as csv_input:
                dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
                non_traffic_tweets = [line[0] for line in dataset]

            tweets = {
                'data': traffic_tweets + non_traffic_tweets,
                'target': [True] * len(traffic_tweets) + [False] * len(non_traffic_tweets),
            }

            training_vectors = count_vect.fit_transform(tweets['data'])

            scores = cross_val_score(clf, training_vectors, tweets['target'], cv=cv)
            # print(scores)
            accuracy = scores.mean()
            print("Accuracy: {} (+/- {})".format(accuracy, scores.std() * 2))

            scores = cross_val_score(clf, training_vectors, tweets['target'], cv=cv, scoring='precision')
            # print(scores)
            precision = scores.mean()
            print("Precision: {} (+/- {})".format(precision, scores.std() * 2))

            scores = cross_val_score(clf, training_vectors, tweets['target'], cv=cv, scoring='recall')
            # print(scores)
            recall = scores.mean()
            print("Recall: {} (+/- {})".format(recall, scores.std() * 2))

            scores = cross_val_score(clf, training_vectors, tweets['target'], cv=cv, scoring='f1')
            # print(scores)
            f1 = scores.mean()
            print("F1: {} (+/- {})".format(f1, scores.std() * 2))

            results.append((calculation, threshold, accuracy, recall, precision, f1))

with open(join(dirname(__file__), 'svm_10folds.csv'), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for result in results:
        csv_writer.writerow(result)