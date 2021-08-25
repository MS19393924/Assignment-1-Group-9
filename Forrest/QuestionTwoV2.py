
def multi():
    wine = np.genfromtxt("winequality-red.csv", delimiter=';', skip_header=1)
    X = wine[:,:11]
    y= wine[:,11:12]
    print(X)
    print(y)
    zero = 0;
    one = 0;
    two = 0;
    for i in range(0, len(y)):
        if y[i] <= 3.33:
            y[i] = 0.0
            zero= zero + 1
        elif y[i] <= 6.67:
            y[i] = 1.0
            one = one + 1
        else:
            y[i] = 2.0
            two = two + 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    LR = LogisticRegression(solver="liblinear")
    LR.fit(X_train,y_train.ravel())
    y_pred = LR.predict(X_test)
    report = classification_report(y_test, y_pred,labels=[0,1,2])
    print(report)
    plot_confusion_matrix(LR,X_test, y_test)
    plt.show()
            
    
    return 


def Binary():
    wine = np.genfromtxt("winequality-red.csv", delimiter=';',  skip_header=1)
    X = wine[:,:11]
    y = wine[:,11:12]
    for i in range(0, len(y)):
        if y[i] <= 5.0:
            y[i] = 0.0
        else:
            y[i] = 1.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    LR = LogisticRegression(solver="liblinear")
    LR.fit(X_train,y_train.ravel())
    y_pred = LR.predict(X_test)
    report = classification_report(y_test, y_pred,labels=[0,1])
    print(report)
    plot_confusion_matrix(LR,X_test, y_test)
    plt.show()
    
    return

def getFreqNums():
    names = []
    values = []
    wine = np.genfromtxt("winequality-red.csv", delimiter=';',  skip_header=1)
    y = wine[:,11:12]
    for i in range(0, 10):
        names.append(str(i))
        values.append(np.count_nonzero(y == i))

    print(names)
    plt.figure(figsize=(9, 10))

    plt.bar(names, values)

    plt.suptitle('frequency of numbers')
    plt.show()
    return

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

#Binary()
#multi()
#getFreqNums()