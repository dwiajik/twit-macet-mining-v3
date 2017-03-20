from nltk import ngrams

def ngrams_tokenizer(tweet, n=1):
    return [' '.join(tupl) for tupl in list(ngrams(tweet.split(), n))]

def tokenize_tweets(labeled_tweets, n=1):
    return [(ngrams_tokenizer(tweet, n), category) for (tweet, category) in labeled_tweets]
