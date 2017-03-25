'''
This script is intended to evaluate dataset using SVM and 10-folds cross validation
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

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/clean/distinct_traffic_tweets.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    traffic_tweets = [(line[0], line[1]) for line in dataset]

with open(os.path.join(os.path.dirname(__file__), 'tweets_corpus/clean/distinct_non_traffic_tweets.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    non_traffic_tweets = [(line[0], line[1]) for line in dataset]

random.shuffle(traffic_tweets)
random.shuffle(non_traffic_tweets)

if sys.argv[1] == "balance":
    traffic_tweets = traffic_tweets[:min([len(traffic_tweets), len(non_traffic_tweets)])]
    non_traffic_tweets = non_traffic_tweets[:min([len(traffic_tweets), len(non_traffic_tweets)])]

labeled_tweets = (traffic_tweets + non_traffic_tweets)
random.shuffle(labeled_tweets)

print('Start analysis with total:', len(labeled_tweets), 'data')
print('Traffic tweets:', len(traffic_tweets),'data')
print('Non traffic tweets:', len(non_traffic_tweets),'data')

times = []
true_positives = []
true_negatives = []
false_positives = []
false_negatives = []
accuracies = []
precisions = []
recalls = []
f_measures = []

for i in range(fold):
    train_set = [(tweet_features(tweet), category) for (tweet, category) in labeled_tweets[0 : i * int(len(labeled_tweets) / fold)]] + \
        [(tweet_features(tweet), category) for (tweet, category) in labeled_tweets[(i + 1) * int(len(labeled_tweets) / fold) : len(labeled_tweets)]]
    test_set = [(tweet_features(tweet), category) for (tweet, category) in labeled_tweets[i * int(len(labeled_tweets) / fold) : (i + 1) * int(len(labeled_tweets) / fold)]]

    print('\nIteration', (i + 1))
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
        if label == 'traffic' and observed == 'non_traffic':
            false_positive += 1
        if label == 'non_traffic' and observed == 'traffic':
            false_negative += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_measure = f1(precision, recall)

    times.append(time)
    true_positives.append(true_positive)
    true_negatives.append(true_negative)
    false_positives.append(false_positive)
    false_negatives.append(false_negative)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f_measures.append(f_measure)

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

print('\nSVM Classifier:')
print('\tAverage training time:', sum(times) / len(times))
print('\tAverage true positive:', sum(true_positives) / len(true_positives))
print('\tAverage true negative:', sum(true_negatives) / len(true_negatives))
print('\tAverage false positives:', sum(false_positives) / len(false_positives))
print('\tAverage false negatives:', sum(false_negatives) / len(false_negatives))
print('\tAverage accuracy:', sum(accuracies) / len(accuracies))
print('\tAverage precision:', sum(precisions) / len(precisions))
print('\tAverage recall:', sum(recalls) / len(recalls))
print('\tAverage F-Measure:', sum(f_measures) / len(f_measures))