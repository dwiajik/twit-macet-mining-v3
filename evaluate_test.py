'''
This script is intended to evaluate dataset using test set
'''

import collections
import csv
import datetime
import json
import os
import random
import re
import sys
import time as t

import nltk
import nltk.classify
from nltk.metrics import scores
from sklearn.svm import LinearSVC

from modules import cleaner, tokenizer

fold = 10

def tweet_features(tweet):
    features = {}
    tweet = cleaner.clean(tweet)

    for word in tweet.split():
        features["{}".format(word)] = tweet.count(word)

    return features

def f1(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))

with open(os.path.join(os.path.dirname(__file__), 'result/generated_datasets/dice/0.1/traffic.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    traffic_tweets = [(tweet_features(line[0]), line[1]) for line in dataset]

with open(os.path.join(os.path.dirname(__file__), 'result/generated_datasets/dice/0.1/non_traffic.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    non_traffic_tweets = [(tweet_features(line[0]), line[1]) for line in dataset]

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/test_set_10000.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    test_set = [(tweet_features(line[0]), line[1]) for line in dataset]

train_set = (traffic_tweets + non_traffic_tweets)
random.shuffle(train_set)

print('Training data:', len(train_set), 'data')
print('Test data:', len(test_set), 'data')

# SVM
start_time = t.time()
svm_classifier = nltk.classify.SklearnClassifier(LinearSVC(max_iter=10000)).train(train_set)
time = round(t.time() - start_time, 2)
accuracy = nltk.classify.accuracy(svm_classifier, test_set)
 
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
for i, (feature, label) in enumerate(test_set):
    observed = svm_classifier.classify(feature)
    if label == 'traffic' and observed == 'traffic':
        true_positive += 1
    if label == 'non_traffic' and observed == 'non_traffic':
        true_negative += 1
    if label == 'non_traffic' and observed == 'traffic':
        # print('{}: {}'.format('fp',feature))
        false_positive += 1
    if label == 'traffic' and observed == 'non_traffic':
        # print('{}: {}'.format('fn',feature))
        false_negative += 1

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f_measure = f1(precision, recall)

print('SVM Classifier:')
print('\t', 'Training time:', time)    
print('\t', 'True positive:', true_positive)
print('\t', 'True negative:', true_negative)
print('\t', 'False positive:', false_positive)
print('\t', 'False negative:', false_negative)
print('\t', 'Accuracy:', accuracy)
print('\t', 'Precision:', precision)
print('\t', 'Recall:', recall)
print('\t', 'F-Measure:', f_measure)