#!/usr/bin/python

import warnings
import sys
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=DeprecationWarning) 

total = len(sys.argv)
cmdargs = str(sys.argv)

for i in xrange(1, total):
	X, y = load_svmlight_file("../data/" + str(sys.argv[i]))
	for curC in [0.0001, 0.001, 0.01, 0.1, 1., 10.]:
		print("==================================================================")
		#print("# File = %s" % (str(sys.argv[i])))
		#print("loss = SqrHingeLoss,")
		#print("reg = SqrL2Reg,")
		#print("solver = DualCoordinateDescent,")
		#print("tol = %f," % 1e-10)
		print("C = %f," % curC)

		svc = LinearSVC(C = curC, tol=1e-10, max_iter=50000, verbose=1, loss="l1")

		svc.fit(X, y)
		results = svc.predict(X)
		accuracy = accuracy_score(y, results)
		print("Accuracy = {}".format(accuracy))
		print("")
