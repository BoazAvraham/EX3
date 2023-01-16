import matplotlib.pyplot as plt


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
