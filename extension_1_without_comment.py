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
import operator
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

    num_hyp = len(hypDict[1])


    #Y = np.empty([refLen * num_hyp, 1], dtype='S500')
    Y = np.empty([refLen * num_hyp, 1], dtype='f')

    bleu = [0.0, 0.0, 0.0, 0.0]
    for idx in xrange(1, len(hypDict) + 1):
        for refIdx in xrange(4):
            for hypIdx in xrange(4):
                #print hypDict[idx][hypIdx], refDict[idx][refIdx]
                if hypDict[idx][hypIdx] == 'n/a':
                    bleu[hypIdx] = bleu[hypIdx] + 0
                else:
                    bleu[hypIdx] = bleu[hypIdx] + smoothedBLEU(hypDict[idx][hypIdx], refDict[idx][refIdx])

        bleu = [float(item / 4) for item in bleu]
        #target_idx, target_val = max(enumerate(bleu), key=operator.itemgetter(1))

        Y[4 * (idx - 1) , 0] = bleu[0]
        Y[4 * (idx - 1) + 1, 0] = bleu[1]
        Y[4 * (idx - 1) + 2, 0] = bleu[2]
        Y[4 * (idx - 1) + 3, 0] = bleu[3]

    #print Y
    #Create Y-label estimated by highest Smoothed-BLEUSCORE
    return Y


def parseFile(path):
    file = open(path)
    txt = file.readlines()

    header = txt[0]
    #print header.split("\t")

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
    #print header
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
            workerFeatureDict[key][2] = 0.5

        if workerFeatureDict[key][5] == 'UNKNOWN':
            workerFeatureDict[key][5] = int(EnglishYrCtr)

        if workerFeatureDict[key][6] == 'UNKNOWN':
            workerFeatureDict[key][6] = int(UrduYrCtr)


    # for key in workerFeatureDict.keys():
    #     print workerFeatureDict[key]
    #['WorkerID', 'IsEnglishNative', 'IsUrduNative', 'LocationIndia', 'LocationPakistan', 'YearSpeakingEnglish', 'YearSpeakingUrdu']
    return workerFeatureDict

def edDistRecursiveMemo(x, y, memo=None):
    ''' A version of edDistRecursive with memoization.  For each x, y we see, we
        record result from edDistRecursiveMemo(x, y).  In the future, we retrieve
        recorded result rather than re-run the function. '''
    if memo is None: memo = {}
    if len(x) == 0:
        return len(y)
    if len(y) == 0:
        return len(x)
    if (len(x), len(y)) in memo:
        return memo[(len(x), len(y))]
    delt = 1 if x[-1] != y[-1] else 0
    diag = edDistRecursiveMemo(x[:-1], y[:-1], memo) + delt
    vert = edDistRecursiveMemo(x[:-1], y, memo) + 1
    horz = edDistRecursiveMemo(x, y[:-1], memo) + 1
    ans = min(diag, vert, horz)
    memo[(len(x), len(y))] = ans
    return ans


def parsePostEdited(filepath):
    file = open(filepath)
    txt = file.readlines()

    header = txt[0].split("\t")
    for item in txt:
        item = item.split("\t")
        #print item[3], item[12], item[21], item[30]

    print header

def parseRank(filepath):
    file = open(filepath)
    txt = file.readlines()

    header = txt[0].split("\t")
    for item in txt:
        item = item.split("\t")
    print header

def parseTestFile(filepath):
    file = open(filepath)
    txt = file.readlines()
    header = txt[0].split()

    sourceDict = {}
    candidateDict = {}
    workerDict = {}
    #[sourceDict, referenceDict, candidateDict, workerDict]
    for i in xrange(1, len(txt)):
        row = txt[i].split("\t")
        sourceDict[i] = row[1] #Urdu
        candidateDict[i] = row[2:6]
        workerDict[i] = row[6:]

    answerList = [sourceDict, candidateDict, workerDict]
    return answerList



def trainAndformatML(Y_train, train_refDict, train_candidateDict, train_workerDict, workerFeatureDict, sourceDict, testcandidateDict, testworkerDict):
    print "Implementing"
    # print Y_train.size
    # print len(train_workerDict.keys()) #358 in train
    # print len(workerFeatureDict.keys()) #51 from the data

    # = np.empty([refLen * num_hyp, 1], dtype='S500')
    num_examples = Y_train.size

    #worker_id - key dictionary to easily index the feature construction.
    workerID = {}
    ctr = 1
    for key in workerFeatureDict.keys():
        workerID[key] = ctr
        ctr += 1

    print "Start Train"
    X_train = np.empty([num_examples, 59], dtype='f') #58 features; 51 writer, 7 other

    #Construct Train Feature
    #print len(sourceDict) 1792
    #print len(train_workerDict.keys()) 358

    list = train_candidateDict[1]
    #temp = train_candidateDict[1].split()
    totalTokenSet = set()
    for strList in list:
        for token in strList.split():
            totalTokenSet.add(token)

    #print train_workerDict[0]

    for i in xrange(1, len(train_workerDict.keys()) + 1):
        totalSet = set()
        list = train_candidateDict[i]
        for str in list:
            for token in str.split():
                totalSet.add(token)

        for j in xrange(4):
            X_train[4 * (i - 1) + j, 0] = len(train_candidateDict[i][j]) / len(sourceDict[i])  #length proportional to source

            target_set = set(train_candidateDict[i][j].split())
            X_train[4 * (i - 1) + j, 1] = len(target_set.intersection(totalSet)) / len(totalSet) # check # of unique elements in each sentence
            worker_id = train_workerDict[i][j].rstrip()

            if worker_id == 'n/a':
                for x in xrange(1, 7):
                    X_train[4 * (i - 1) + j, x + 1] = 0


            else:
                X_train[4 * (i - 1) + j, 2] = workerFeatureDict[worker_id][1]   #Native English? 0/1
                X_train[4 * (i - 1) + j, 3] = workerFeatureDict[worker_id][2]   #Native Urdu? 0/1
                X_train[4 * (i - 1) + j, 4] = workerFeatureDict[worker_id][3]   #Live in India? 0/1
                X_train[4 * (i - 1) + j, 5] = workerFeatureDict[worker_id][4]   #Live in Pakistan? 0/1
                X_train[4 * (i - 1) + j, 6] = workerFeatureDict[worker_id][5]   #Year Speaking English?
                X_train[4 * (i - 1) + j, 7] = workerFeatureDict[worker_id][6]   #Year Speaking Urdu?

            #Worker ID assignment
            try:
                update_idx = workerID[worker_id]
                X_train[4 * (i - 1) + j, 7 + update_idx] = 1
                #print update_idx, worker_id
                #print X_train[4*(i-1)+j,]
            except KeyError:
                pass
                #update_idx = None #If update_idx == None, leave all column 0


    print ("Start Adaboost Training")
    #X_train = np.float()
    Y_train = np.ravel(Y_train)
    clf = AdaBoostRegressor(n_estimators=1000, loss='exponential')
    clf.fit(X_train, Y_train)

    #NEED TO TEST
    #RE-FORMAT THE TEST DATA
    #print len(sourceDict) = 1792
    X_test = np.empty([4 * len(sourceDict), 59], dtype='f')


    #print len(train_workerDict.keys())
    for i in xrange(1, len(sourceDict.keys()) + 1):
        totalSet = set()
        list = testcandidateDict[i]
        for str in list:
            for token in str.split():
                totalSet.add(token)

        for j in xrange(4):
            X_test[4 * (i - 1) + j, 0] = len(testcandidateDict[i][j]) / len(sourceDict[i]) # length proportional to sentence length

            target_set = set(testcandidateDict[i][j].split())
            X_test[4 * (i - 1) + j, 1] = len(target_set.intersection(totalSet)) / len(totalSet) #check # of unique elements
            worker_id = testworkerDict[i][j].rstrip()

            if worker_id == 'n/a':
                for x in xrange(1, 7):
                    X_test[4 * (i - 1) + j, x + 1] = 0

            else:
                X_test[4 * (i - 1) + j, 2] = workerFeatureDict[worker_id][1]   #Native English? 0/1
                X_test[4 * (i - 1) + j, 3] = workerFeatureDict[worker_id][2]   #Native Urdu? 0/1
                X_test[4 * (i - 1) + j, 4] = workerFeatureDict[worker_id][3]   #Live in India? 0/1
                X_test[4 * (i - 1) + j, 5] = workerFeatureDict[worker_id][4]   #Live in Pakistan? 0/1
                X_test[4 * (i - 1) + j, 6] = workerFeatureDict[worker_id][5]   #Year Speaking English?
                X_test[4 * (i - 1) + j, 7] = workerFeatureDict[worker_id][6]   #Year Speaking Urdu?

            #Worker ID Assignment
            try:
                update_idx = workerID[worker_id]
                X_test[4 * (i - 1) + j, 7 + update_idx] = 1

            except KeyError:
                pass

    #START TESTING
    print "Testing"
    Y_predicted = clf.predict(X_test)

    solutionList = [0 for _ in xrange(len(sourceDict))]

    print len(solutionList)

    for i in xrange(len(sourceDict)):
        maxVal = max(Y_predicted[4 * i : 4 * (i + 1)])
        maxIdx = [idx for idx, val in enumerate(Y_predicted[4 * i : 4 * (i + 1)]) if val == maxVal]
        solutionList[i] = (maxVal, maxIdx[0] + 4 * i)


    output = open("extension_output.txt", "a")
    for i in xrange(1, len(solutionList) + 1):
        #solutionIdx = solutionList[i][1]
        #print i, solutionList[i][1], solutionList[i][1] - 4*i
        #print solutionList[i-1], solutionList[i-1][1]%4
        idx = solutionList[i-1][1] % 4
        #print i + "||" + solutionList[i] +"||"+ solutionList[i][1]-3*(i-1) +"||"+4*i
        #print i, testcandidateDict[i][idx]
        output.write((testcandidateDict[i][idx]) + '\n')

    output.close()




if __name__ == "__main__":
    print "Start"
    filepath = os.path.relpath('data/train_translations.tsv') #actual data
    parsedDataList = parseFile(filepath)
    #[sourceDict, referenceDict, candidateDict, workerDict]
    train_refDict = parsedDataList[1]
    train_candidateDict = parsedDataList[2]
    train_workerDict = parsedDataList[3]

    #CONSTRUCT BLEUDICT for train purpose
    Y_train = constructBLEUDict(train_refDict, train_candidateDict)

    #WORKER FEATURE
    filepath = os.path.relpath('data/survey.tsv') #worker feature
    workerFeatureDict = constructWorkerFeature(filepath)

    #TEST DICT
    print "TEST"
    filepath = os.path.relpath('data/test_translations.tsv')
    parsedTestData = parseTestFile(filepath)
    entire_sourceDict = parsedTestData[0]
    test_candidateDict = parsedTestData[1]
    test_workerDict = parsedTestData[2]
    #[sourceDict, candidateDict, workerDict]

    print "Train and Test"
    trainAndformatML(Y_train, train_refDict, train_candidateDict, train_workerDict, workerFeatureDict, entire_sourceDict, test_candidateDict, test_workerDict)

    print "Complete"
