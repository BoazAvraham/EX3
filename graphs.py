import matplotlib.pyplot as plt

import ExpectiMax
import numpy as np


def generate_conf_mat(topics, expectiMax):
    con_mat = np.array((ExpectiMax.CLUSTERSIZE, len(topics)))
    for doc in expectiMax.docs:
        selected_topic = np.argmax(expectiMax.wti[doc.index]) # the cluster index with the highest probability
        for topic in doc.topics:
            con_mat[selected_topic.index][topic.index] += 1
    # for i in range(ExpectiMax.CLUSTERSIZE):
    #     for topic in range(len(topics)):
    #         con_mat[i][topic] =
def save_log_l_g(iteration, log_l):
    # save log l graph
    plt.plot(iteration, log_l)
    plt.xlabel("iteration number")
    plt.ylabel("log likelihood")
    plt.title("log likelihood per iteration")
    plt.show()
    plt.savefig('log_l_per_iter.png')


def save_perplex(iteration, perplexity):
    # save log perplexity
    plt.plot(iteration, perplexity)
    plt.xlabel("iteration number")
    plt.ylabel("perplexity")
    plt.title("perplexity per iteration")
    plt.show()
    plt.savefig('perplexity_per_iter.png')
