from __future__ import division
import optparse
import sys
import nltk
import numpy as np
import sklearn
import math
import timeit
from collections import Counter
import chardet
import os

def parseFile(path):
    file = open(path)
    txt = file.readlines()

    header = txt[0]
    header = header[:2] + header[6:]
    print header.split("\t")

    referenceDict = {}
    candidateDict = {}
    sourceDict = {}
    workerDict = {}
    sentenceDict = {}

    tempList = txt[1].split("\t")

    for i in xrange(1, len(txt)):
        sentenceList = txt[i].split("\t")
        sourceDict[i] = sentenceList[1] #URDU
        referenceDict[i] = sentenceList[2:6]
        candidateDict[i] = sentenceList[6:10]
        workerDict[i] = sentenceList[11:]
        updatedSentenceList = sentenceList[:2] + sentenceList[6:]
        sentenceDict[i] = updatedSentenceList

    write = open('test_translations.tsv', 'ab')
    write.write(header)
    for key in sentenceDict.keys():
        mylist = "\t".join(sentenceDict[key])
        write.write(mylist)
    write.close()



if __name__ == "__main__":
    #THIS FILE GENERATES "test_translations.tsv" which does not contain reference sentences.
    filepath = os.path.relpath('data/translations.tsv') #actual data
    parsedDataList = parseFile(filepath)
