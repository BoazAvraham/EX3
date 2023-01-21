# Yuval Ronen    Boaz Avraham     205380132   203668132
import sys
from collections import Counter

import ExpectiMax
from getAllwords import getAllWordsPerDoc, get_all_topics
from getAllwords import getAllWords
from graphs import save_log_l_g, save_perplex

import numpy as np


class Details():
    def __init__(self):
        self.developmentSetFileName = sys.argv[1]
        self.topics_file_name = sys.argv[2]


class Sets():
    def __init__(self, all_words_per_doc):
        self.docs_histograms = [Counter(x) for x in all_words_per_doc]


def writeToOutput():
    output = open(details.outputFileName, 'w')
    output.write("#Yuval Ronen\tBoaz Avraham\t205380132\t203668132")
    for i in range(1, 30):
        output.write("\n" + "#Output" + str(i) + "\t" + str(details.output[i]))
    output.close()


def start_to_train(algo):
    # init variables for table
    prev_Py = algo.Py
    iteration = [i for i in range(30)]
    log_l = []
    perplex = []
    iter = 0

    # calculate pik
    algo.Pik()

    while prev_Py < 0.1:
        algo.start_training()
        prev_Py = algo.Py
        algo.Py = algo.Y_teta()  # calculate new y_teta
        if algo.Py < prev_Py:
            raise Exception("the Likelihood decrease, we have a bug!")
        print('new P(y|teta) = ' + str(algo.Py))

        # graph details
        log_l.append(algo.Py)
        perplex.append(algo.perplexity(algo.Py))
        iter += 1
        if iter == len(iteration):
            break

    # save graphs
    save_log_l_g(iteration, log_l)
    save_perplex(iteration, perplex)




if __name__ == '__main__':
    # input:  < development set filename > <topics file name>
    details = Details()

    #### 1 Init
    allwords, topicsPerDoc = getAllWordsPerDoc(details.developmentSetFileName)
    all_doc_words = getAllWords(details.developmentSetFileName)
    mixed_histograms = [Counter(allwords[i]) for i in range(len(allwords))]

    all_topics = get_all_topics(details.topics_file_name)
    algo = ExpectiMax.ExpectationMaximizationSmoothed(mixed_histograms, allwords, all_doc_words, topicsPerDoc)
    # start train
    start_to_train(algo)
    # confusion mat
    conf_mat = algo.generate_conf_mat(all_topics)
    algo.accuracy(conf_mat,all_topics)

    # con_mat = np.array((ExpectiMax.CLUSTERSIZE, len(all_topics)))
    # for i in range(ExpectiMax.CLUSTERSIZE):
    #     for topic