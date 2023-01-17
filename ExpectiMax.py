import math
from collections import Counter
import numpy as np
import pandas as pd


class Document:

    def __init__(self, histogram, doc_size, topics):
        self.histogram = histogram
        self.words_count = len(histogram)
        self.topics = topics
        self.doc_size = doc_size
        self.Zi_list = []


class Cluster:

    def __init__(self, index, wti, docs_size):
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


K = 10
eps = 0.001
LAMBDA = 0.06  # from ex2
CLUSTERSIZE = 9
RARE = 3  # threshold for filtering rare words


class ExpectationMaximizationSmoothed:

    def __init__(self, mixed_docs_histograms, allwords, all_doc_words, all_docs_topics):
        self.all_words_hist = Counter(all_doc_words)
        self.all_words = self.all_words_hist.keys()
        self.docs = [Document(mixed_docs_histograms[i], len(allwords[i]), all_docs_topics[i]) for i in range(len(mixed_docs_histograms))]
        self.delete_rare_words()

        self.vocab_size = len(self.all_words_hist)

        # this is the prob matrix. at the beginning the probability is 1 for every doc t in cluster %t
        self.wti = np.zeros((len(self.docs), CLUSTERSIZE))
        for t in range(len(self.docs)):
            self.wti[t][t % CLUSTERSIZE] = 1

        self.pik = np.array([{} for i in range(CLUSTERSIZE)])

        self.alpha = np.array([(sum(self.wti[t][i] for t in range(len(self.docs))) /
                                len(self.docs)) for i in range(CLUSTERSIZE)])  # in the beginning its uniform

        self.Py = - math.inf

    def delete_rare_words(self):
        """In order to reduce time and place complexity you should
            filter rare words. A rare word, for this exercise, is a word that occurs 3 times or less in
            the input corpus """
        all_words_items = list(self.all_words_hist.items())
        # delete from all words histogram
        for word, count in all_words_items:
            if count <= RARE:
                del self.all_words_hist[word]
        # delete words not in  all words histogram
        for doc in self.docs:
            doc_keys = list(doc.histogram.keys())
            for word in doc_keys:
                if word not in self.all_words_hist:
                    doc.doc_size -= doc.histogram[word] # update doc size
                    del doc.histogram[word]

    def generate_conf_mat(self, topics):
        con_mat = np.zeros((CLUSTERSIZE, len(topics) + 1))
        for doc_i in range(len(self.docs)):
            selected_topic = np.argmax(self.wti[doc_i])  # the cluster index with the highest probability
            con_mat[selected_topic][len(topics)] += 1  # number of articles that were assigned to this cluster
            for topic in self.docs[doc_i].topics:
                con_mat[selected_topic][topics.index(topic)] += 1  # Mij= num of articles in jth topic in ith cluster
        topics += ['cluster_size']
        data = {topic: np.flip(con_mat[:, topics.index(topic)]) for topic in topics}
        df = pd.DataFrame(data, index=[str(i) for i in range(CLUSTERSIZE - 1, -1, -1)])
        df = df.convert_dtypes() # convert float to int
        df.to_csv('con_matrix.csv')
        print(df)
        return con_mat

    def accuracy(self, con_mat):
        # accuracy
        sum_max = 0
        for i in range(CLUSTERSIZE):
            sum_max += np.argmax(con_mat[i][:CLUSTERSIZE])
        print(sum_max / len(self.docs))

    def lidProbability(self, c_x, S):
        """c_x- number of occurrences of the event in set. S is the set size.
         V= vocab size"""
        return (c_x + LAMBDA) / (S + LAMBDA * self.vocab_size)

    def perplexity(self, logLike):
        ## TODO not sure about the the number of words in the perplexity
        return math.exp((-1 / self.vocab_size) * logLike)


    def start_training(self):
        # E step
        # calculate wti
        for t in range(len(self.docs)):
            self.docs[t].Zi_list = [self.Zi(t, i) for i in range(CLUSTERSIZE)]
            max_zi = max(self.docs[t].Zi_list)
            self.wti[t] = [self.Xi_Yt(i, t, max_zi) for i in
                           range(CLUSTERSIZE)]  # for every doc yt we calculate p(xi|yt)

        # M step

        # calculate new alpha i with given wti
        self.alpha_i()
        # calculate new pik with given wti
        self.Pik()

    def alpha_i(self):
        for i in range(CLUSTERSIZE):
            # 1/N * ∑w_ti
            candidate = sum((self.wti[t][i] for t in range(len(self.docs)))) / len(self.docs)
            # Smoothing in the M step
            self.alpha[i] = candidate if candidate > eps else eps
        # need to make sure that sum( αi) = 1. so αj' = αj / sum(αi)
        s = np.sum(self.alpha)
        # calculation of p(x_i)
        self.alpha = self.alpha / s

    def Pik(self):
        for i in range(CLUSTERSIZE):
            denominator = sum((self.docs[t].doc_size * self.wti[t][i]) for t in range(len(self.docs)))
            lid_smooth_den = denominator + self.vocab_size * LAMBDA
            for t in range(len(self.docs)):
                for k in self.docs[t].histogram:
                    if k in self.pik[i].keys(): # if pik[i][k] exists. we need to add to it
                        self.pik[i][k] += self.docs[t].histogram[k] * self.wti[t][i] / lid_smooth_den
                    else:
                        self.pik[i][k] = LAMBDA / lid_smooth_den
                        self.pik[i][k] += (self.docs[t].histogram[k] * self.wti[t][i]) / lid_smooth_den

    def Zi(self, t, i):
        ws = self.docs[t].histogram  # this is the histogram of doc number t

        # in the log we ignore 0 because we applied lidstone.
        right_hand = sum((ws[w] * np.log(self.pik[i][w]) for w in ws))

        left_hand = np.log(self.alpha[i])
        return right_hand + left_hand

    def Xi_Yt(self, i, t, m):
        '''this is w_t_i,
        i is the num of cluster, t is the num of doc, m is max(zi)'''
        if self.docs[t].Zi_list[i] - m < -K:
            return 0
        exp = lambda i: np.exp(self.docs[t].Zi_list[i] - m)
        denominator = sum((exp(j) for j in range(CLUSTERSIZE) if self.docs[t].Zi_list[j] - m >= -K))
        return exp(i) / denominator

    def lan_sum_Ezi(self, t):
        sum_zis = 0
        max_zi = max(self.docs[t].Zi_list)
        for i in range(CLUSTERSIZE):
            power = self.docs[t].Zi_list[i] - max_zi
            if power >= -K:
                sum_zis += np.exp(power)
        return max_zi + np.log(sum_zis)

    def Y_teta(self):
        ln_L = sum((self.lan_sum_Ezi(t) for t in range(len(self.docs))))
        return ln_L


if __name__ == '__main__':
    ExpectationMaximizationSmoothed()
# okay decompiling D:\MASTER_DEGREE\RESTORED\ExpectiMax.cpython-37.pyc
