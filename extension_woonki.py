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


def constructBLEUDict(refDict, hypDict):
    bleuDict = {}
    main_keys = refDict.keys()

    refLen = len(refDict.keys())
    hypLen = len(hypDict.keys())

    # maxlen = 0
    # for i in xrange(1, len(hypDict) + 1):
    #     temp = hypDict[i]
    #     temp1 = len(temp[0])
    #     temp2 = len(temp[1])
    #     temp3 = len(temp[2])
    #     temp4 = len(temp[3])
    #
    #     tempmax = max(temp1,temp2,temp3,temp4)
    #     maxlen = max(maxlen, tempmax)
    # print maxlen
    Y = np.empty([refLen, 1], dtype='S500')

    ctr = 0
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    bleu4 = 0
    bleu = [0.0, 0.0, 0.0, 0.0]
    for idx in xrange(1, len(hypDict) + 1):
        for refIdx in xrange(4):
            for hypIdx in xrange(4):
                print hypDict[idx][hypIdx], refDict[idx][refIdx]
                bleu[hypIdx] = bleu[hypIdx] + smoothedBLEU(hypDict[idx][hypIdx], refDict[idx][refIdx])

        bleu = [float(item / 4) for item in bleu]
        print bleu, idx



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
        workerDict[i] = sentenceList[10:]

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
    EnglishYrCtr = 0
    UrduYrCtr = 0

    for i in xrange(1, len(txt)):
        row = txt[i].split()
        id = row[0]

        #Process of encoding YES into 1, while NO into 0
        row = [1 if item =='YES' else item for item in row]
        row = [0 if item == 'NO' else item for item in row]

        #Calculating Mean value for #ofYrs Speaking English and Urdu
        if row[5] == 'UNKNOWN':
            pass
        else:
            EnglishYrCtr += int(row[5])

        if row[6] == 'UNKNOWN':
            pass
        else:
            UrduYrCtr += int(row[6])

        workerFeatureDict[id] = row

    EnglishYrCtr = EnglishYrCtr / (len(txt) - 1)
    UrduYrCtr = UrduYrCtr / (len(txt) - 1)



    #Manually fixing obviously wrong value
    workerFeatureDict['a2iouac3vzbks6'][5] = int(EnglishYrCtr)
    workerFeatureDict['a2iouac3vzbks6'][6] = int(UrduYrCtr)

    #DATA PREPROCESSING
    #UPDATE two unknown values to 0.5 for whether that person is native in English or Urdu
    #UPDATE UNKNOWN yrs of speaking language with mean value
    for key in workerFeatureDict.keys():
        if workerFeatureDict[key][1] == 'UNKNOWN':
            workerFeatureDict[key][1] = 0.5

        if workerFeatureDict[key][2] == 'UNKNOWN':
            workerFeatureDict[key][1] = 0.5

        if workerFeatureDict[key][5] == 'UNKNOWN':
            workerFeatureDict[key][5] = int(EnglishYrCtr)

        if workerFeatureDict[key][6] == 'UNKNOWN':
            workerFeatureDict[key][6] = int(UrduYrCtr)


    # for key in workerFeatureDict.keys():
    #     print workerFeatureDict[key]
    #['WorkerID', 'IsEnglishNative', 'IsUrduNative', 'LocationIndia', 'LocationPakistan', 'YearSpeakingEnglish', 'YearSpeakingUrdu']
    return workerFeatureDict






if __name__ == "__main__":
    print "Start"
    filepath = os.path.relpath('data/train_translations.tsv') #actual data
    parsedDataList = parseFile(filepath)
    #[sourceDict, referenceDict, candidateDict, workerDict]
    refDict = parsedDataList[1]
    candidateDict = parsedDataList[2]


    # for i in xrange(1, len(candidate) + 1):
    #     print refDict[i]
    #     print candidateDict[i]

    #CONSTRUCT BLEUDICT
    constructBLEUDict(refDict, candidateDict)


    #worker feature
    filepath = os.path.relpath('data/survey.tsv') #worker feature
    workerFeatureDict = constructWorkerFeature(filepath)

    #filepath = os.path.relpath('data/train_translations.tsv')

