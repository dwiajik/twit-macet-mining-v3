from math import *
from decimal import Decimal

def vector(a, b):
    vec_a, vec_b = {}, {}
    for token in set(a + b):
        vec_a[token], vec_b[token] = 0, 0
    for token in a:
        vec_a[token] += 1
    for token in b:
        vec_b[token] += 1
    return list(set(a + b)), vec_a, vec_b

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

class Jaccard:
    def unique(self, a):
        return list(set(a))

    def intersect(self, a, b):
        return list(set(a) & set(b))

    def union(self, a, b):
        return list(set(a) | set(b))

    def index(self, a = [], b = []):
        try:
            return len(self.intersect(a, b)) / len(self.union(a, b))
        except ZeroDivisionError as e:
            return 1

# same performance as original jaccard
class WeightedJaccard:
    def index(self, a, b):
        tokens, vec_a, vec_b = vector(a, b)
        numerator = sum(list(map(lambda token: min(vec_a[token], vec_b[token]), tokens)))
        denominator = sum(list(map(lambda token: max(vec_a[token], vec_b[token]), tokens)))
        try:
            return numerator / denominator
        except ZeroDivisionError as e:
            return 1

# same performance as original jaccard
class ExtendedJaccard:
    def index(self, a, b):
        tokens, vec_a, vec_b = vector(a, b)
        sop = sum(list(map(lambda token: vec_a[token] * vec_b[token], tokens)))
        dot_a = sum([value * value for attr, value in vec_a.items()])
        dot_b = sum([value * value for attr, value in vec_b.items()])

        denominator = dot_a + dot_b - sop
        try:
            return sop / denominator
        except ZeroDivisionError as e:
            return 1

class Cosine:
    def index(self, a, b):
        tokens, vec_a, vec_b = vector(a, b)
        # sum of product
        sop = sum(list(map(lambda token: vec_a[token] * vec_b[token], tokens)))
        # square root of sum of a square
        sqrt_soas = sqrt(sum([value * value for attr, value in vec_a.items()]))
        # square root of sum of b square
        sqrt_sobs = sqrt(sum([value * value for attr, value in vec_b.items()]))
        try:
            return sop / (sqrt_soas * sqrt_sobs)
        except ZeroDivisionError as e:
            return 1

class Dice:
    def index(self, a, b):
        tokens, vec_a, vec_b = vector(a, b)
        sop = sum(list(map(lambda token: vec_a[token] * vec_b[token], tokens)))
        dot_a = sum([value * value for attr, value in vec_a.items()])
        dot_b = sum([value * value for attr, value in vec_b.items()])
        try:
            return (2 * sop) / (dot_a + dot_b)
        except ZeroDivisionError as e:
            return 1

# the values are weird
class Euclidean:
    def index(self, a, b):
        tokens, vec_a, vec_b = vector(a, b)
        subtract = list(map(lambda token: abs(vec_a[token] - vec_b[token]), tokens))
        sop = sum([e * e for e in subtract])
        distance = sqrt(sop)
        return 1 / (1 + distance)

class Manhattan:
    def index(self, a, b):
        tokens, vec_a, vec_b = vector(a, b)
        sum_of_subtract = sum(list(map(lambda token: abs(vec_a[token] - vec_b[token]), tokens)))
        return sum_of_subtract
        # try:
        #     return 1 - (sum_of_subtract / len(tokens))
        # except ZeroDivisionError as e:
        #     return 1

# complicated
# http://simeon.wikia.com/wiki/Minkowski_distance
class Minkowski:
    def index(self, a, b):
        return 0

# not suitable
class Matching:
    def index(self, a, b):
        return 0

class Overlap:
    def intersect(self, a, b):
        return list(set(a) & set(b))

    def index(self, a = [], b = []):
        try:
            return len(self.intersect(a, b)) / min(len(a), len(b))
        except ZeroDivisionError as e:
            return 1

# the values are weird
class Pearson:
    def index(self, a, b):
        tokens, vec_a, vec_b = vector(a, b)
        sop = sum(list(map(lambda token: vec_a[token] * vec_b[token], tokens)))
        sum_of_a = sum([value for attr, value in vec_a.items()])
        sum_of_b = sum([value for attr, value in vec_b.items()])
        sum_of_square_a = sum([value * value for attr, value in vec_a.items()])
        sum_of_square_b = sum([value * value for attr, value in vec_b.items()])
        try:
            numerator = sop - ((sum_of_a * sum_of_b) / len(tokens))
            denominator = sqrt((sum_of_square_a - (sum_of_a ** 2 / len(tokens))) * (sum_of_square_b - (sum_of_b ** 2 / len(tokens))))
            return numerator / denominator
        except ZeroDivisionError as e:
            return 1

class Combination:
    def __init__(self):
        self.cosine = Cosine()
        self.dice = Dice()
        self.jaccard = Jaccard()
        self.manhattan = Manhattan()
        self.overlap = Overlap()

    def index(self, a, b):
        cosine = self.cosine.index(a, b)
        dice = self.dice.index(a, b)
        jaccard = self.jaccard.index(a, b)
        manhattan = self.manhattan.index(a, b)
        overlap = self.overlap.index(a, b)

        return mean([cosine, dice, jaccard, manhattan, overlap])
