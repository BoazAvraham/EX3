#Yuval Ronen    Boaz Avraham     205380132   203668132
from collections import Counter

FILTER_BELOW = 3

def getAllWords(file):
    """read the file and compose a list of all the words in file
    words can appear more than one time"""
    readFile = open(file, 'r')
    allWords = []
    for line in readFile:
        values = line.split()
        if len(values) != 0 and values[0] != "<TRAIN" and values[0] != "<TEST":
            # we want to skip the lines of the subjects and empty lines
            allWords += values
    return allWords


def getAllWordsPerDoc(file):
    """read the file and compose a list of all the words in file per doc
    words can appear more than one time"""
    counter = Counter(getAllWords(file))
    filter_set = {pair[0] for pair in counter.items() if pair[1] <= FILTER_BELOW}

    readFile = open(file, 'r')
    wordsPerDoc = []
    for line in readFile:
        values = line.split()
        doc_index = -1
        if len(values) != 0:
            if values[0] != "<TRAIN":
                #wordsPerDoc[doc_index][:] = [x for x in wordsPerDoc[doc_index] if x not in filter_set]
                wordsPerDoc[doc_index] += values
            else:
                doc_index += 1
                wordsPerDoc.append([])


    return wordsPerDoc