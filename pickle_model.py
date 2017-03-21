import sys

from sklearn.svm import LinearSVC

from modules import cleaner, tokenizer

def tweet_features(tweet):
    features = {}
    tweet = cleaner.clean(tweet)

    for word in tweet.split():
        features["{}".format(word)] = tweet.count(word)

    return features

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

train_set = [(tweet_features(tweet), category) for (tweet, category) in labeled_tweets]

svm_classifier = nltk.classify.SklearnClassifier(LinearSVC(max_iter=10000)).train(train_set)

with open(sys.argv[1], 'wb') as output:
    pickle.dump(svm_classifier, output, pickle.HIGHEST_PROTOCOL)