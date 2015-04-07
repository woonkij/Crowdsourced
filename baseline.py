import optparse
from smoothedBLEU import smoothedBLEU
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

optparser = optparse.OptionParser()
optparser.add_option("-r", "--readTranslationFiles", dest="input", default="data/translations.tsv", help="Read Translation Files.")
optparser.add_option("-s", "--readSurvey", dest='survey', default='data/survey.tsv')
(opts, _) = optparser.parse_args()

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
# print workerFeatures


print "Building data set...\n"

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
				try: scores[j] += smoothedBLEU(candidate, reference)
				except: pass

	best_idx[i] = np.argmax(scores)

	feats[i] = []
	for ID in workerDict[i]:
		feats[i] += workerFeatures[ID]



############### Classification

y = np.array(best_idx.values())
X = np.array(feats.values())
imp = Imputer(missing_values='NaN', strategy='most_frequent', copy=False)
X = imp.fit_transform(X)

X_train, y_train = X[0:358,], y[0:358,]

print "Training random forest...\n"

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, y_train)

predictions = rf.predict(X)


################# write output to file

write = open("baseline_output.txt", 'w')

for (i, p) in zip(xrange(1, len(candidateDict)+1), predictions):
    write.write(str(candidateDict[i][p]) + '\n')
write.close()






