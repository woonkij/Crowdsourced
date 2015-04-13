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
from sklearn.ensemble import AdaBoostRegressor


def smoothedBLEU(hypothesis, reference):
    updateVar = 1.0
    K = 5.0

    ngram_Weight_Vector = [0 for _ in xrange(4)]

    #SMOOTHING_4
    #http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    h_ngramList = [0 for _ in xrange(4)]
    for n in xrange(1, 5): #1,2,3,4
        h_ngrams = Counter([tuple(hypothesis[i:i + n]) for i in xrange(len(hypothesis) + 1 - n)])
        r_ngrams = Counter([tuple(reference[i:i + n]) for i in xrange(len(reference) + 1 - n)])

        num_match = max([sum((h_ngrams & r_ngrams).values()), 0])

        #number of tokens for translation phrase
        h_ngramList[n - 1] = sum(h_ngrams.values())

        if num_match == 0:
            #print math.log(len(hypothesis))
            updateVar = updateVar * K / (math.log(len(hypothesis)))
            ngram_Weight_Vector[n - 1] = float(1.0 / updateVar)
        else:
            ngram_Weight_Vector[n - 1] = num_match

    #SMOOTHING_5
    new_weight_vec = [0 for _ in xrange(5)]
    new_weight_vec[0] = ngram_Weight_Vector[0] + 1
    ngram_Weight_Vector.append(0) #for updating purpose: last = 0

    for i in xrange(1, 5): #1,2,3,4
        new_weight_vec[i] = (new_weight_vec[i - 1] + ngram_Weight_Vector[i - 1] + ngram_Weight_Vector[i]) / 3

    #pop initial dummy value
    new_weight_vec.remove(new_weight_vec[0])
    score = 0
    for i in xrange(4):
        #print h_ngramList
        score = score + 0.25*math.log(new_weight_vec[i] / h_ngramList[i])
    score = math.exp(score)
    BrevityPenalty = min(1, math.exp(1 - len(reference)/len(hypothesis)))
    score = score * BrevityPenalty
    return score



def parseFile(path):
    file = open(path)
    txt = file.readlines()

    header = txt[0]
    print header.split("\t")

    referenceDict = {}
    candidateDict = {}
    sourceDict = {}
    workerDict = {}

    tempList = txt[1].split("\t")

    for i in xrange(1, len(txt)):
        sentenceList = txt[i].split("\t")
        sourceDict[i] = sentenceList[1] #URDU
        referenceDict[i] = sentenceList[2:6]
        candidateDict[i] = sentenceList[6:10]
        workerDict[i] = sentenceList[11:]

    answerList = [sourceDict, referenceDict, candidateDict, workerDict]
    return answerList


def constructWorkerFeature(path):
    file = open(path)
    txt = file.readlines()

    header = txt[0].split()
    print header

    #number of translators = 51
    #thus, the features are 51 + 6(native eng, native urdu, location india, location paki, yrsEnglish, yrsUrdu)

    #return the dictionary regarding the information of the worker
    workerFeatureDict = {}

    for i in xrange(1, len(txt)):
        row = txt[i].split()
        workerFeatureDict[row[0]] = row

    for key in workerFeatureDict.keys():
        print workerFeatureDict[key]


def constructBLEUFeature(path):
    print "Constructing Feature with BLEU"
    file = open(path)
    txt = file.readlines()

    referenceDict = {}
    candidateDict = {}
    sourceDict = {}
    workerDict = {}

    header = txt[0].split("\t")
    print header

    for i in xrange(1, len(txt)):
        sentenceList = txt[i].split("\t")
        sourceDict[i] = sentenceList[1] #URDU
        referenceDict[i] = sentenceList[2:6]
        candidateDict[i] = sentenceList[6:10]
        workerDict[i] = sentenceList[11:]

    answerList = [sourceDict, referenceDict, candidateDict, workerDict]

    X_BleuFeature = np.empty([len(txt) - 1, 4], dtype='f')
    #print X_BleuFeature.shape

    for i in xrange(1, len(txt)):
        ref1 = referenceDict[i][0]
        ref2 = 





if __name__ == "__main__":
    print "Start"
    #filepath = os.path.relpath('data/train_translations.tsv') #actual data
    #parsedDataList = parseFile(filepath)


    #worker feature
    #filepath = os.path.relpath('data/survey.tsv') #worker feature
    #constructWorkerFeature(filepath)

    filepath = os.path.relpath('data/train_translations.tsv')
    constructBLEUFeature(filepath)