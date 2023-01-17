#Yuval Ronen    Boaz Avraham     205380132   203668132
from collections import Counter

FILTER_BELOW = 3

def get_all_topics(file_name):
    readFile = open(file_name, 'r')
    all_topics = []
    for line in readFile:
        values = line.split()
        if len(values) != 0:
            # we want to skip the lines of the subjects and empty lines
            all_topics += values
    # for the confusion matrix
    return all_topics


def getAllWords(file):
    """read the file and compose a list of all the words in file
    words can appear more than one time"""
    readFile = open(file, 'r')
    allWords = []
    for line in readFile:
        values = line.split()
        if len(values) != 0 and values[0] != "<TRAIN":
            # we want to skip the lines of the subjects and empty lines
            allWords += values
    return allWords


def getAllWordsPerDoc(file):
    """read the file and compose a list of all the words in file per doc
    words can appear more than one time"""
    readFile = open(file, 'r')
    wordsPerDoc = []
    topicsPerDoc = []
    for line in readFile:
        values = line.replace('>\n', '\n')
        values = values.split()
        if len(values) != 0:
            if values[0] != "<TRAIN":
                wordsPerDoc.append(values) # add all the wordsin doc to list
            else:
                topicsPerDoc.append(values[2:]) #add all the topics in docs per list
    return wordsPerDoc, topicsPerDoc