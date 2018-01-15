# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 22:35:32 2018

@author: alien
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,2:len(dataset.iloc[0,])-1].values
y = dataset.iloc[:,len(dataset.iloc[0,])-1].values

#FeatureScaling
from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
sc_X = SC.fit_transform(x)

#Split dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sc_X, y, test_size = 0.25, random_state = 0)

#fiting decision tree
from sklearn.linear_model import LogisticRegression
classifier =  LogisticRegression()
classifier.fit(x_train, y_train);

y_pred = classifier.predict(x_test);

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



#Visualizing the Dataset and the Decision Boundary
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#Visualizing Test set
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

del(i,j)