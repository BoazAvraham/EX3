#Yuval Ronen    Boaz Avraham     205380132   203668132
import sys
from collections import Counter

import ExpectiMax
from getAllwords import getAllWordsPerDoc
from getAllwords import getAllWords

from myInit import myInit


class Details():
    def __init__(self):
        self.developmentSetFileName = sys.argv[1]
        self.languageVocabularySize = 300000


class Sets():
    def __init__(self, all_words_per_doc):
        self.docs_histograms = [Counter(x) for x in all_words_per_doc]


def writeToOutput():
    output = open(details.outputFileName, 'w')
    output.write("#Yuval Ronen\tBoaz Avraham\t205380132\t203668132")
    for i in range(1, 30):
        output.write("\n" + "#Output" + str(i) + "\t" + str(details.output[i]))
    output.close()


if __name__ == '__main__':
    # input:  < development set filename >
    details = Details()
    developmentSetFileName = details.developmentSetFileName

    #### 1 Init
    allwords = getAllWordsPerDoc(details.developmentSetFileName)
    all_doc_words = getAllWords(details.developmentSetFileName)
    mixed_histograms = [Counter(allwords[i]) for i in range(len(allwords))]
    algo = ExpectiMax.ExpectationMaximizationSmoothed(mixed_histograms, all_doc_words)

