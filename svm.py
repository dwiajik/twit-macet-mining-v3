import csv
from os.path import dirname, join, exists
from random import shuffle
import resource
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

from modules.cleaner import clean

## Initialize categories and count vectorizer
categories = ['traffic', 'non_traffic']
count_vect = CountVectorizer(preprocessor=clean)

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

            ## Open training set
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

            # print(training_vectors.shape)
            # print(len(tweets['target']))

            # start_time = time.clock()
            clf = LinearSVC(max_iter=10000).fit(training_vectors, tweets['target'])
            # training_time = round(time.clock() - start_time, 2) / 100
            # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

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

            results.append((calculation, threshold, accuracy, prfs[0][0], prfs[1][0], prfs[2][0]))
            # for doc, category in zip(test_doc, predicted):
            #     print('%r => %s' % (doc, categories[category]))

with open(join(dirname(__file__), 'svm_test.csv'), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for result in results:
        csv_writer.writerow(result)