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

class Overlap:
    def intersect(self, a, b):
        return list(set(a) & set(b))

    def index(self, a = [], b = []):
        try:
            return len(self.intersect(a, b)) / min(len(a), len(b))
        except ZeroDivisionError as e:
            return 1
