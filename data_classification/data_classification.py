from __future__ import print_function
import pandas as pd
import numpy as np
from numpy import *
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('train.csv')
trainx = data.iloc[:, :-1].values
trainy = data.iloc[:, 128].values
data = pd.read_csv('test_raw.csv')
testx = data.iloc[:, :128].values
testy = np.zeros(len(testx))

print (len(testx))
print (len(trainx))

X_train = trainx
y_train = trainy
X_test = testx
y_test = testy
clf = RandomForestClassifier(n_estimators=100)
clf.fit(trainx, trainy)
testy = clf.predict(X_test)

mu1 = []
mu2 = []
for index in range(7910):
    st = str(index+6000)
    st1 = "ID_"
    len1 = 8 - len(st)
    for i in range(len1):
        st1 += "0"
    st1 += st
    mu1.append(st1)

print (len(mu1))
for index in range(len(testy)):
    mu2.append("cls_" + str(testy[index]))
dataframe = pd.DataFrame({'ID': mu1, 'Pred': mu2})
dataframe.to_csv("test2.csv", index=False, sep=',')

