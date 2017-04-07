'''
This script is intended to evaluate dataset using SVM and 10-folds cross validation in n times.
'''

import collections
import csv
import datetime
import json
import os
import random
import re
import statistics
import sys
import time as t

import nltk
import nltk.classify
from nltk.metrics import scores
from sklearn.svm import LinearSVC

from modules import cleaner, tokenizer

n = 30
folds = 10

def tweet_features(tweet):
    features = {}
    tweet = cleaner.clean(tweet)

    for word in tweet.split():
        features["{}".format(word)] = tweet.count(word)

    return features

def f1(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))

with open(os.path.join(os.path.dirname(__file__), 'result/generated_datasets/overlap/0.7/traffic.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    traffic_tweets = [(line[0], line[1]) for line in dataset]

with open(os.path.join(os.path.dirname(__file__), 'result/generated_datasets/overlap/0.7/non_traffic.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    non_traffic_tweets = [(line[0], line[1]) for line in dataset]

# random.shuffle(traffic_tweets)
# random.shuffle(non_traffic_tweets)

# if sys.argv[1] == "balance":
#     traffic_tweets = traffic_tweets[:min([len(traffic_tweets), len(non_traffic_tweets)])]
#     non_traffic_tweets = non_traffic_tweets[:min([len(traffic_tweets), len(non_traffic_tweets)])]

cv_times = []
cv_true_positives = []
cv_true_negatives = []
cv_false_positives = []
cv_false_negatives = []
cv_accuracies = []
cv_precisions = []
cv_recalls = []
cv_f_measures = []

for x in range(n):
    labeled_tweets = (traffic_tweets + non_traffic_tweets)
    random.shuffle(labeled_tweets)

    times = []
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    accuracies = []
    precisions = []
    recalls = []
    f_measures = []

    for i in range(folds):

        train_set = [(tweet_features(tweet), category) for (tweet, category) in labeled_tweets[0 : i * int(len(labeled_tweets) / folds)]] + \
            [(tweet_features(tweet), category) for (tweet, category) in labeled_tweets[(i + 1) * int(len(labeled_tweets) / folds) : len(labeled_tweets)]]
        test_set = [(tweet_features(tweet), category) for (tweet, category) in labeled_tweets[i * int(len(labeled_tweets) / folds) : (i + 1) * int(len(labeled_tweets) / folds)]]

        print('\rn:  {}/{}\tfolds:  {}/{} '.format(x + 1, n, i + 1, folds), end='')

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

    cv_times.append(statistics.mean(times))
    cv_true_positives.append(statistics.mean(true_positives))
    cv_true_negatives.append(statistics.mean(true_negatives))
    cv_false_positives.append(statistics.mean(false_positives))
    cv_false_negatives.append(statistics.mean(false_negatives))
    cv_accuracies.append(statistics.mean(accuracies))
    cv_precisions.append(statistics.mean(precisions))
    cv_recalls.append(statistics.mean(recalls))
    cv_f_measures.append(statistics.mean(f_measures))

print('\nSVM Classifier:')
print('\tAverage training time: {}'.format(statistics.mean(cv_times)))
print('\tAverage true positive: {}'.format(statistics.mean(cv_true_positives)))
print('\tAverage true negative: {}'.format(statistics.mean(cv_true_negatives)))
print('\tAverage false positives: {}'.format(statistics.mean(cv_false_positives)))
print('\tAverage false negatives: {}'.format(statistics.mean(cv_false_negatives)))
print('\tAverage accuracy: {}'.format(statistics.mean(cv_accuracies)))
print('\tAverage precision: {}'.format(statistics.mean(cv_precisions)))
print('\tAverage recall: {}'.format(statistics.mean(cv_recalls)))
print('\tAverage F-Measure: {}'.format(statistics.mean(cv_f_measures)))