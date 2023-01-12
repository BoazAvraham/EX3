
import math
from collections import defaultdict, Counter
import numpy as np

class Document:

    def __init__(self, histogram, cluster, index):
        self.histogram = histogram
        self.words_count = len(histogram)
        self.cluster = cluster
        self.index = index
        self.Zi_list = []

    def __len__(self):
        return len(self.histogram)


class Cluster:

    def __init__(self, index):
        self.histogram = None
        self.prob = None
        self.index = index
        self.docs_in_cluster = set()

    def add(self, docs):
        self.docs_in_cluster.add(docs)

    def update_histogram(self):
        self.histogram = Counter()
        for c in self.docs_in_cluster:
            self.histogram |= c.histogram

    def size(self):
        return len(self.histogram)

    def set_prob(self, p):
        self.prob = p

    def __getitem__(self, arg):
        return self.histogram[arg]

    def __len__(self):
        return len(self.histogram)


K = 10
eps = 0.001

class ExpectationMaximizationSmoothed:

    def __init__(self, mixed_docs_histograms):
        self.docs = []
        self.clusters = [Cluster(i) for i in range(9)]
        self.Py = math.inf
        for i in range(len(mixed_docs_histograms)):
            cluster = self.clusters[i % 9]
            new_doc = Document(mixed_docs_histograms[i], cluster, i)
            self.docs.append(new_doc)
            cluster.add(new_doc)

        for x in self.clusters:
            x.update_histogram()

        prev_Py = math.inf
        while prev_Py > 100:
            self.start_training()
            prev_Py = self.Py
            self.Py = self.Y_teta()
            print('new P(y|teta) = ' + str(self.Py))

    def start_training(self):
        probs = dict()
        for t in range(len(self.docs)):
            self.docs[t].Zi_list = [self.Zi(t, i) for i in range(len(self.clusters))]
            max_zi = max(self.docs[t].Zi_list)
            probs[t] = [self.Xi_Yt(i, t, max_zi) for i in range(len(self.clusters))]

        for x in self.clusters:
            candidate = sum((probs[wti][x.index] for wti in probs)) / len(self.docs)
            x.prob = candidate if candidate > eps else eps

        s = sum((x.prob for x in self.clusters))
        for x in self.clusters:
            x.prob = x.prob / s

        for x in self.clusters:
            x.update_histogram()

    def Xi(self, i):
        return len(self.clusters[i]) / len(self.docs)

    def Zi(self, t, i):
        ws = self.docs[t].histogram
        x = self.clusters[i]
        right_hand = sum((ws[w] * np.log(x[w] / len(x)) for w in ws if x[w] > 0))
        left_hand = np.log(self.Xi(i))
        return right_hand + left_hand

    def Xi_Yt(self, i, t, m):
        if self.docs[t].Zi_list[i] - m > K:
            return 0
        exp = lambda i: np.exp(self.docs[t].Zi_list[i] - m)
        denominator = sum((exp(j) for j in range(9) if self.docs[t].Zi_list[j] - m >= -K))
        return exp(i) / denominator

    def lan_sum_Ezi(self, t):
        sum_zis = 0
        max_zi = max(self.docs[t].Zi_list)
        T = len(self.docs[t])
        print(self.docs[t].Zi_list)
        for i in range(len(self.clusters)):
            power = pow(self.docs[t].Zi_list[i], T) - pow(max_zi, T)
            if power > K:
                sum_zis += np.exp(power)

        return pow(max_zi, T) + sum_zis

    def Y_teta(self):
        ln_L = sum((self.lan_sum_Ezi(t) for t in range(len(self.docs))))
        return ln_L


if __name__ == '__main__':
    ExpectationMaximizationSmoothed()
# okay decompiling D:\MASTER_DEGREE\RESTORED\ExpectiMax.cpython-37.pyc
