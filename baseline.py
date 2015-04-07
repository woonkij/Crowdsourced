#!/usr/bin/env python

import optparse, json, sys
from smoothedBLEU import smoothedBLEU
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

optparser = optparse.OptionParser()
optparser.add_option("-r", "--readTranslationFiles", dest="input", default="data/translations.tsv", help="Read Translation Files.")
optparser.add_option("-s", "--readSurvey", dest='survey', default='data/survey.tsv')
optparser.add_option("-d", "--baselineData", dest='data', default=None, help="Data for learning algorithms")
(opts, _) = optparser.parse_args()


if opts.data is None:

	# substitute strings in survey.tsv
	sub = {'YES': 1, 'NO': 0, 'UNKNOWN': float('NaN')}

	f_survey = open(opts.survey)
	txt = f_survey.read().splitlines()
	workerFeatures = {}
	for i in xrange(1, len(txt)):
		t = txt[i].split('\t')
		# key: workerID, value: features
		workerFeatures[t[0]] = [sub[x] if x in sub.keys() else int(x) for x in t[1:]]
	f_survey.close()
	workerFeatures['n/a'] = [float('NaN') for _ in xrange(6)]
	# sys.stderr.write(workerFeatures,'\n')


	sys.stderr.write("Building data set...\n")

	file = open(opts.input)
	txt = file.read().splitlines()

	sourceDict = {}
	referenceDict = {}
	candidateDict = {}
	workerDict = {}

	for i in xrange(1, len(txt)): # ignore header (0)
	    sentenceList = txt[i].split("\t")
	    sourceDict[i] = sentenceList[1] #Urdu
	    referenceDict[i] = sentenceList[2:6] #LDC sentences
	    candidateDict[i] = sentenceList[6:10] #Turkers translated
	    workerDict[i] = sentenceList[10:]

	best_idx = {} # labels for supervised training
	feats = {} # corresponding worker-level features

	######### build dataset

	for i in xrange(1, len(candidateDict) + 1):
		scores = [0.0 for _ in candidateDict[i]]

		for (j, candidate) in enumerate(candidateDict[i]):
			if candidate != 'n/a':
				for reference in referenceDict[i]:
					if reference != 'n/a':
						try: scores[j] += smoothedBLEU(candidate, reference)
						except: pass

		best_idx[i] = np.argmax(scores)

		feats[i] = []
		for ID in workerDict[i]:
			feats[i] += workerFeatures[ID]

	with open(r'data/data.json', 'w') as fp:
		json.dump([feats, best_idx, candidateDict], fp, default=str)

else:
	with open(r'data/data.json', 'rb') as fp:
		feats, best_idx, candidateDict = json.load(fp)




############### Classification

y = np.array( map(int, best_idx.values()) )
X = np.array(feats.values())

# fill in missing/unknown values
imp = Imputer(missing_values='NaN', strategy='most_frequent', copy=False)
X = imp.fit_transform(X)

# 20% used for training data, as specified in proposal
X_train, y_train = X[0:358,], y[0:358,]

sys.stderr.write("Training classifier(s)...\n")

np.random.seed(1)

rf = RandomForestClassifier(n_estimators = 150)
rf.fit(X_train, y_train)

# svm_param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
# ]

# svc = GridSearchCV( SVC(), svm_param_grid )
# svm = svc.fit(X_train, y_train).best_estimator_


# predicts index of best translation (0,1,2, or 3)
predictions = rf.predict(X)




################# write output to file

# out = open("baseline_output.txt", 'w')

if opts.data is None:
	for (i, p) in enumerate(predictions):
		sys.stdout.write(candidateDict[i + 1][p] + '\n')
else:
	for (i, p) in enumerate(predictions):
		sys.stdout.write(candidateDict[str(i + 1)][p].encode('utf-8') + '\n')

# out.close()



# out = open("oracle_output.txt", 'w')

# # 'best' translations corresponding to best_idx computed above
# for (i, c) in enumerate(y):
# 	out.write(candidateDict[i + 1][c] + '\n')
# out.close()






