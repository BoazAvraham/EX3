#Yuval Ronen    Boaz Avraham     205380132   203668132

import math
from collections import defaultdict

class LidstoneModel:
    def __init__(self, details, sets):
        self.S = sets.trainingSetSize
        self.X = 300000 # same as ex2
        self.details = details
        self.sets = sets
        self.start_training()

    # def start_training(self):
    #     lam_values = [x/100 for x in range(1, 201)]
    #     self.perplexityDict = { lam: self.perplexity(lam) for lam in lam_values}
    #     self.best_lamda = min(self.perplexityDict, key=self.perplexityDict.get)

    def probability(self, c_x, lam):
        """c_x- number of occurrences of the event in set. S is the set size.
         X= number of events in set"""
        return float(c_x + lam) / (self.S + lam * self.X)

    # def perplexity(self, lam,  test_set = None):
    #     logSum = 0
    #     if test_set == None:
    #         test_set = self.sets.validationSet
    #     for w in test_set:
    #         p_w = self.probability(self.sets.trainingSetWordsCounter[w], lam)
    #         logSum += math.log(p_w)
    #
    #     return math.exp(-logSum / len(test_set))