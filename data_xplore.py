# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:53:50 2019

@author: keert
"""

# Import relevant libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

phish_data = pd.read_csv("FIU_Phishing_Mitre_Dataset.csv")

phish_data["update_age(months)"] = phish_data["update_age(days)"].values/30
phish_data.head()

num_data = phish_data.iloc[:,[0,1,2,6]].values
num_lab = phish_data["Label"].values

col_1 = num_data[:,0]
create = num_data[:,1]
expiry = num_data[:,2]
update = num_data[:,3]


plt.subplot(2, 2, 1)
plt.plot(col_1, '.')

plt.xlabel('sample')
plt.ylabel('col_1')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(create, '.')

plt.xlabel('sample')
plt.ylabel('create')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(expiry, '.')

plt.xlabel('sample')
plt.ylabel('expiry')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(update, '.')

plt.xlabel('sample')
plt.ylabel('update')
plt.grid(True)


plt.figure()
plt.subplot(2, 2, 1)
plt.hist(col_1, bins = 50)

plt.xlabel('sample')
plt.ylabel('HIST - col_1')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.hist(create, bins = 50)

plt.xlabel('sample')
plt.ylabel('HIST - create')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.hist(expiry, bins = 50)

plt.xlabel('sample')
plt.ylabel('HIST - expiry')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.hist(update, bins = 50)

plt.xlabel('sample')
plt.ylabel('HIST - update')
plt.grid(True)


num_data_scaled = StandardScaler().fit_transform(num_data)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=500),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X_train, X_test, y_train, y_test = train_test_split(num_data_scaled, num_lab, test_size = 0.2)

f1_vals = []

for name, clf in zip(names, classifiers):
    
    print(name)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    y_hat = clf.predict(X_test)
    
    f1_vals.append(f1_score(y_test, y_hat))
    
    print("Done")
    
print("Done")