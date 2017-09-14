import csv
import numpy
import math
from copy import deepcopy

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [x for x in dataset[i]]
	return dataset

def writeCsv(filename, mydict):
	f = open(filename, 'wt')
	try:
		writer = csv.writer(f)
		for key in mydict:
			writer.writerow([key] + mydict[key])
	finally:
		f.close()

wpbc_raw = loadCsv('wpbc.csv')
wdbc_raw = loadCsv('wdbc.csv')

#print(wpbc_raw[1])
#print(wdbc_raw[1])

wpbc_data = {}
wdbc_data = {}
for line in wpbc_raw:
	wpbc_data[line[0]] = line[1:]
	pass

for line in wdbc_raw:
	wdbc_data[line[0]] = line[1:]
	pass

labeled = {}
unlabeled = {}
for key in wdbc_data:
	if (key in wpbc_data):
		labeled[key] = [wpbc_data[key][0]] + wdbc_data[key]
	elif (key[:-2] in wpbc_data):
		labeled[key] = [wpbc_data[key[:-2]][0]] + wdbc_data[key]
	elif (key[:-1] in wpbc_data):
		labeled[key] = [wpbc_data[key[:-1]][0]] + wdbc_data[key]
	else:
		unlabeled[key] = wdbc_data[key]

#for key in labeled:
#	print(labeled[key][0:2])
print(len(labeled))
print(len(wdbc_data))

writeCsv('unlabeled.csv', unlabeled)

xTrain = []
yTrain = []

addingColumn1 = []
addingColumn2 = []
for key in wpbc_data:
	val = wpbc_data[key];
	yTrain.append(val[0] == 'R')
	tmp = [float(x) for x in val[1:-1]]
	addingColumn1.append(float(val[-2]));
	addingColumn2.append(float(val[1]));
	"""for i in xrange(19, 24):
		tmp[i] = -math.log(tmp[i])"""
	"""if (val[-1] == '?'):
		tmp.append(0.0);
	else:
		tmp.append(float(val[-1]))"""
	xTrain.append(tmp)

#print(xTrain)
#print(wpbc_data)
#print(wdbc_data)
xUnlabel = [];
for key in unlabeled:
	val = unlabeled[key];
	xUnlabel.append( [sum(addingColumn2)/len(addingColumn2)] + [float(x) for x in val[1:]] + [sum(addingColumn1)/len(addingColumn1)])

#from sklearn import datasets
#iris = datasets.load_iris()
#print(iris.data)
#print(iris.target)

from sklearn.naive_bayes import GaussianNB
import sklearn
gnb = GaussianNB()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()


from sklearn.preprocessing import normalize
xTotal = xTrain + xUnlabel
xTotal = normalize(xTotal)
#xTrain = normalize(xTrain)
#print(len(xTrain))
xTrain = xTotal[0:198]
xUnlabel = xTotal[198:]
#print(xTrain.shape)
#print(xUnlabel.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=6)

pca.fit(xTotal)
print(pca.explained_variance_ratio_)
xTrain = pca.transform(xTrain)
xUnlabel = pca.transform(xUnlabel)
#xTotal = pca.transform(xTotal)

NFolder = 10
chunkSize = math.ceil(1.0 * len(yTrain)/NFolder)
y_predTrain = []
for i in xrange(10):
	startIdx = int(i*chunkSize);
	endIdx = int(min((i+1)*chunkSize, len(yTrain)))
	xTestTmp = xTrain[startIdx:endIdx]
	yTestTmp = yTrain[startIdx:endIdx]
	xTrainTmp = numpy.concatenate([xTrain[0:startIdx], xTrain[endIdx:]])
	yTrainTmp = numpy.append(yTrain[0:startIdx], yTrain[endIdx:])
	sw = [(float(x) - 0.5) * 0 + 0.5 for x in yTrainTmp];
	sw = numpy.asarray(sw)
	y_predTrain = numpy.append(y_predTrain, gnb.fit(xTrainTmp, yTrainTmp, sw).predict(xTestTmp))
#y_predUnlabeled = gnb.fit(xTrain, yTrain).predict(xUnlabel)

"""y_Total = deepcopy(yTrain)
for y in y_predUnlabeled:
	y_Total.append(y)
y_predTrain = gnb.fit(xTotal, y_Total).predict(xTrain)"""


truetrue = zip(y_predTrain, yTrain)
print(sum(a == b for a, b in truetrue))
print(int(sum(a and b for a, b in truetrue)))

#from sklearn.ensemble import AdaBoostClassifier
