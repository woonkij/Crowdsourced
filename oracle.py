#!/usr/bin/env python
import optparse
import sys
import math
import smoothedBLEU

optparser = optparse.OptionParser()
optparser.add_option("-r", "--readTranslationFiles", dest="input", default="data/translations.tsv", help="Read Translation Files.")
(opts, _) = optparser.parse_args()

file = open(opts.input)
txt = file.readlines()

referenceDict = {}
candidateDict = {}
sourceDict = {}
workerDict = {}

for i in xrange(1, len(txt)):
    sentenceList = txt[i].split("\t")
    sourceDict[i] = sentenceList[1] #Urdu
    referenceDict[i] = sentenceList[2:6] #LDC sentences
    candidateDict[i] = sentenceList[6:10] #Turkers translated
    workerDict[i] = sentenceList[11:]

answerList = [sourceDict, referenceDict, candidateDict, workerDict]


#START DEFAULT OUTPUT
#write = open("output.txt", "a")

for i in xrange(1, len(candidateDict) + 1):
    refList  = []
    for idx in xrange(4):
        refList.append(referenceDict[i][idx])

    candidateList = []
    for idx in xrange(4):
        candidateList.append(candidateDict[i][idx])

    scoreList = [0 for _ in xrange(4)]

    for candidate_idx in xrange(4):
        for ref_idx in xrange(4):


            try:
                scoreList[candidate_idx] += smoothedBLEU.smoothedBLEU(candidateList[candidate_idx], refList[ref_idx])
            except ZeroDivisionError:
                print refList[ref_idx], "REF"
                print candidateList[candidate_idx], "candidate"


            # if candidateList[candidate_idx] == 'n/a':
            #     scoreList[candidate_idx] += 0
            # else:
            #     scoreList[candidate_idx] += smoothedBLEU.smoothedBLEU(candidateList[candidate_idx], refList[ref_idx])


    print scoreList
