import csv
from os.path import dirname, join

from sklearn.cluster import Birch
from sklearn.feature_extraction.text import CountVectorizer

from modules.cleaner import clean

count_vect = CountVectorizer(preprocessor=clean)

with open(join(dirname(__file__), 'result/generated_datasets/traffic.csv'), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    traffic_tweets = [line[0] for line in dataset]

print(len(traffic_tweets))
traffic_vectors = count_vect.fit_transform(traffic_tweets)

brc = Birch(branching_factor=50, n_clusters=None, threshold=2.5, compute_labels=True)
brc.fit(traffic_vectors)
# predicted = brc.predict(X)
print(len(brc.subcluster_centers_))
print(brc.subcluster_centers_[0])