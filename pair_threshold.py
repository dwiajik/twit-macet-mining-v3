import argparse
import csv
import os
import random

parser = argparse.ArgumentParser(description='Evaluate classifier model using ten folds cross validation.')
parser.add_argument('-o', '--output', default='output.csv', help='File name for output CSV, e.g. output.csv')
args = parser.parse_args()

calculations = [
    'cosine',
    'dice',
    'jaccard',
    'overlap',
    'lcs',
]

cv_accuracies = []
cv_times = []

with open(os.path.join(os.path.dirname(__file__), 'result/diff.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    next(dataset)
    different_tweets = [({
        'cosine': float(line[4]),
        'dice': float(line[5]),
        'jaccard': float(line[6]),
        'overlap': float(line[7]),
        'lcs': float(line[8]),
        }, 'different') for line in dataset]

with open(os.path.join(os.path.dirname(__file__), 'result/sim.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    next(dataset)
    similar_tweets = [({
        'cosine': float(line[4]),
        'dice': float(line[5]),
        'jaccard': float(line[6]),
        'overlap': float(line[7]),
        'lcs': float(line[8]),
        }, 'similar') for line in dataset]

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    accuracies = {
        'cosine': 0,
        'dice': 0,
        'jaccard': 0,
        'overlap': 0,
        'lcs': 0,
    }

    for calculation in calculations:
        tweets = different_tweets + similar_tweets

        results = {}

        true = 0
        false = 0
        for (similarity, category) in tweets:
            if similarity[calculation] >= threshold:
                if category == 'similar':
                    true += 1
                else:
                    false += 1
            else:
                if category == 'different':
                    true += 1
                else:
                    false += 1
        accuracies[calculation] = true / (true + false)

    for calculation in calculations:
        cv_accuracies.append((threshold, calculation, accuracies[calculation]))

with open(os.path.join(os.path.dirname(__file__), args.output), 'a', newline='\n') as csv_output:
    csv_writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for (t, c, acc) in cv_accuracies:
        csv_writer.writerow([t, c, acc])
        print('{}\t{}\t{}'.format(t, c, acc))
            