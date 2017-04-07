import csv
from os.path import dirname, join
from random import shuffle
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
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

with open(join(dirname(__file__), 'tweets_corpus/test_set_10000.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    dataset = [(line[0], line[1]) for line in dataset]
    shuffle(dataset)
    test = {
        'data': [line[0] for line in dataset],
        'target': [categories.index(line[1]) for line in dataset],
    }

tweets = {
    'data': traffic_tweets + non_traffic_tweets,
    'target': [0] * len(traffic_tweets) + [1] * len(non_traffic_tweets),
}

training_vectors = count_vect.fit_transform(tweets['data'])

# print(training_vectors.shape)
# print(len(tweets['target']))

start_time = time.time()
clf = LinearSVC(max_iter=10000).fit(training_vectors, tweets['target'])
training_time = round(time.time() - start_time, 2)

test_vectors = count_vect.transform(test['data'])

predicted = clf.predict(test_vectors)
accuracy = np.mean(predicted == test['target'])

prfs = precision_recall_fscore_support(test['target'], predicted)

print('Training time: {}'.format(training_time))
print('Accuracy: {}'.format(accuracy))
print('Precision: {}'.format(prfs[0][0]))
print('Recall: {}'.format(prfs[1][0]))
print('F-score: {}'.format(prfs[2][0]))

# for doc, category in zip(test_doc, predicted):
#     print('%r => %s' % (doc, categories[category]))