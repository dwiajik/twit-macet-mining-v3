'''
This script is experimental. Intended to evaluate (10-folds cross validation) pickled model.
'''

import collections
import csv
import datetime
import json
import os
import pickle
import random
import re
import sys

import nltk
from sklearn.svm import LinearSVC

from modules import cleaner, tokenizer

def tweet_features(tweet):
    features = {}
    tweet = cleaner.clean(tweet)

    for word in tweet.split():
        features["{}".format(word)] = tweet.count(word)

    return features

def f1(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))

with open('original.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)
    print(type(svm_classifier))

fold = 10

with open(os.path.join(os.path.dirname(__file__), 'distinct_traffic_tweets.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    traffic_tweets = [(line[0], line[1]) for line in dataset]

with open(os.path.join(os.path.dirname(__file__), 'distinct_non_traffic_tweets.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    non_traffic_tweets = [(line[0], line[1]) for line in dataset]

random.shuffle(traffic_tweets)
random.shuffle(non_traffic_tweets)

labeled_tweets = (traffic_tweets + non_traffic_tweets)
random.shuffle(labeled_tweets)

true_positives = []
true_negatives = []
false_positives = []
false_negatives = []
accuracies = []
precisions = []
recalls = []
f_measures = []

for i in range(fold):
    test_set = [(tweet_features(tweet), category) for (tweet, category) in labeled_tweets[i * int(len(labeled_tweets) / fold) : (i + 1) * int(len(labeled_tweets) / fold)]]
    
    # SVM
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

    true_positives.append(true_positive)
    true_negatives.append(true_negative)
    false_positives.append(false_positive)
    false_negatives.append(false_negative)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f_measures.append(f_measure)

    print('SVM Classifier:')
    print('\t', 'True positive:', true_positive)
    print('\t', 'True negative:', true_negative)
    print('\t', 'False positive:', false_positive)
    print('\t', 'False negative:', false_negative)
    print('\t', 'Accuracy:', accuracy)
    print('\t', 'Precision:', precision)
    print('\t', 'Recall:', recall)
    print('\t', 'F-Measure:', f_measure)

print('\nSVM Classifier:')
print('\tAverage true positive:', sum(true_positives) / len(true_positives))
print('\tAverage true negative:', sum(true_negatives) / len(true_negatives))
print('\tAverage false positives:', sum(false_positives) / len(false_positives))
print('\tAverage false negatives:', sum(false_negatives) / len(false_negatives))
print('\tAverage accuracy:', sum(accuracies) / len(accuracies))
print('\tAverage precision:', sum(precisions) / len(precisions))
print('\tAverage recall:', sum(recalls) / len(recalls))
print('\tAverage F-Measure:', sum(f_measures) / len(f_measures))