import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import os
import pandas as pd
import matplotlib as mpl
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV




wine = np.genfromtxt("winequality-red.csv", delimiter=';', skip_header=1)
X = wine[:,:11]
y= wine[:,11:12]
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

ridge = Ridge()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = dict()
grid['alpha'] = arange(0, 1, 0.01)

GSearch = GridSearchCV(ridge, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
results = GSearch.fit(X_train,y_train)
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
scores = cross_val_score(GSearch, GSearch.predict(X_test), y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))





