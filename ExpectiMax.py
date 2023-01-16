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
LAMBDA = 0.06  # from ex2





class ExpectationMaximizationSmoothed:

    def __init__(self, mixed_docs_histograms, all_doc_words):
        self.all_words_hist = Counter(all_doc_words)
        self.all_words = self.all_words_hist.keys()

        self.vocab_size = len(self.all_words_hist)
        self.docs = []
        self.clusters = [Cluster(i) for i in range(9)]
        self.Py = - math.inf
        for i in range(len(mixed_docs_histograms)):
            cluster = self.clusters[i % 9]
            new_doc = Document(mixed_docs_histograms[i], cluster, i)
            self.docs.append(new_doc)
            cluster.add(new_doc)
        # this is the prob matrix. at the begining the probability is 1 for every doc t in cluster %t
        self.wti = np.zeros((len(self.docs), 9))
        self.pik = np.array([{}, {}, {}, {}, {}, {}, {}, {}, {}])
        for t in range(len(self.docs)):
            self.wti[t][t % 9] = 1


        for x in self.clusters:
            x.update_histogram()

        # calculate pik
        self.Pik()
        prev_Py = self.Py
        while prev_Py < 100:
            self.start_training()
            prev_Py = self.Py
            self.Py = self.Y_teta()
            if self.Py < prev_Py:
                raise Exception("the Likelihood decrease, we have a bug!")
            print('new P(y|teta) = ' + str(self.Py))

    def lidProbability(self, c_x, S):
        """c_x- number of occurrences of the event in set. S is the set size.
         V= vocab size"""
        return (c_x + LAMBDA) / (S + LAMBDA * self.vocab_size)

    def start_training(self):
        # probs = dict()
        # E step


        # calculate wti
        for t in range(len(self.docs)):
            self.docs[t].Zi_list = [self.Zi(t, i) for i in range(len(self.clusters))]
            max_zi = max(self.docs[t].Zi_list)
            self.wti[t] = [self.Xi_Yt(i, t, max_zi) for i in
                        range(len(self.clusters))]  # for every doc yt we calculate p(xi|yt)
            # probs[t] = [self.Xi_Yt(i, t, max_zi) for i in
            #             range(len(self.clusters))]  # for every doc yt we calculate p(xi|yt)
        # M step
        for x in self.clusters:
            # 1/N * ∑w_ti
            candidate = sum((self.wti[t][x.index] for t in range(len(self.docs)))) / len(self.docs)
            # candidate = sum((probs[wti][x.index] for wti in probs)) / len(self.docs)
            # Smoothing in the M step
            x.prob = candidate if candidate > eps else eps

        s = sum((x.prob for x in self.clusters))
        # calculation of p(x_i)
        for x in self.clusters:
            x.prob = x.prob / s

        # calculate new pik with given wti
        self.Pik()

        # for x in self.clusters:
        #     x.update_histogram()

    def Pik(self):
        # naive
        # pik_list = []
        for i in range(9):
            denominator = sum((len(self.docs[t]) * self.wti[t][i]) for t in range(len(self.docs)))
            lid_smooth_den = denominator + self.vocab_size * LAMBDA
            for t in range(len(self.docs)):
                for k in self.docs[t].histogram:
                    if k in self.pik[i].keys():
                        self.pik[i][k] += self.docs[t].histogram[k] * self.wti[t][i] / lid_smooth_den
                    else:
                        self.pik[i][k] = LAMBDA / lid_smooth_den
                        self.pik[i][k] += (self.docs[t].histogram[k] * self.wti[t][i]) / lid_smooth_den

            # for k in self.all_words:
            #     nominator = sum((self.docs[t].histogram[k] * self.wti[t][i]) for t in range(len(self.docs)))
            #     lid = self.lidProbability(nominator, denominator)
            #     pik_dict[k] = lid
            # self.pik[i] = pik_dict
        # self.pik = np.array(pik_list)

    def Xi(self, i):
        return len(self.clusters[i]) / len(self.docs)

    def Zi(self, t, i):
        ws = self.docs[t].histogram  # this is the histogram of doc number t
        ws_keys = ws.keys()
        ws_values = np.array(list(ws.values()))
        x = self.clusters[i]  # this is cluster i, x[w] is the combined histogram of cluster i

        # in the log we ignore 0 because we applied lidstone.
        right_hand = sum((ws[w] * np.log(self.pik[i][w]) for w in ws))

        # right_hand = np.sum(np.multiply(ws_values, np.log(pik)))
        # right_hand = sum((ws[w] * np.log(x[w] / len(x)) for w in ws if x[w] > 0))
        left_hand = np.log(self.Xi(i))
        return right_hand + left_hand

    def Xi_Yt(self, i, t, m):
        '''this is w_t_i,
        i is the num of cluster, t is the num of doc, m is max(zi)'''
        if self.docs[t].Zi_list[i] - m < -K:
            return 0
        exp = lambda i: np.exp(self.docs[t].Zi_list[i] - m)
        denominator = sum((exp(j) for j in range(9) if self.docs[t].Zi_list[j] - m >= -K))
        return exp(i) / denominator

    def lan_sum_Ezi(self, t):
        sum_zis = 0
        max_zi = max(self.docs[t].Zi_list)
        # T = len(self.docs[t])
        # print(self.docs[t].Zi_list)
        for i in range(len(self.clusters)):
            # power = self.docs[t].Zi_list[i] - pow(max_zi, t)
            power = self.docs[t].Zi_list[i] - max_zi
            if power >= -K:
                sum_zis += np.exp(power)
        # if sum_zis == 0: # we want to avoid log(0)
        #     return pow(max_zi, t)
        return max_zi + np.log(sum_zis)

    def Y_teta(self):
        ln_L = sum((self.lan_sum_Ezi(t) for t in range(len(self.docs))))
        return ln_L


if __name__ == '__main__':
    ExpectationMaximizationSmoothed()
# okay decompiling D:\MASTER_DEGREE\RESTORED\ExpectiMax.cpython-37.pyc
