import numpy as np
import matplotlib.pyplot as plt 

trainSetX = np.genfromtxt('train.csv', dtype=float, usecols=[0,1,2,3,4,5], delimiter=',');
trainSetY = np.genfromtxt('train.csv', dtype=str, usecols=[6], delimiter=',');

testSetX = np.genfromtxt('test.csv', dtype=float, usecols=[0,1,2,3,4,5], delimiter=',');
testSetY = np.genfromtxt('test.csv', dtype=str, usecols=[6], delimiter=',');

carX = trainSetX[0:23,0];
carY = trainSetX[0:23,3];
internetX = trainSetX[24:47,0];
internetY = trainSetX[24:47,3];
runningX = trainSetX[48:71,0];
runningY = trainSetX[48:71,3];
stairsX = trainSetX[72:95,0];
stairsY = trainSetX[72:95,3];
walkingX = trainSetX[96:119,0];
walkingY = trainSetX[96:119,3];

carXT = testSetX[0:9,0];
carYT = testSetX[0:9,3];
internetXT = testSetX[10:19,0];
internetYT = testSetX[10:19,3];
runningXT = testSetX[20:29,0];
runningYT = testSetX[20:29,3];
stairsXT = testSetX[30:39,0];
stairsYT = testSetX[30:39,3];
walkingXT = testSetX[40:49,0];
walkingYT = testSetX[40:49,3];

plt.figure();
car = plt.scatter(carX, carY, c='black');
internet = plt.scatter(internetX, internetY, c='r');
running = plt.scatter(runningX, runningY, c='g');
stairs = plt.scatter(stairsX, stairsY, c='b');
walking = plt.scatter(walkingX, walkingY, c='y');

cart = plt.scatter(carXT, carYT, c='black', marker='+');
internett = plt.scatter(internetXT, internetYT, c='r', marker='+');
runningt = plt.scatter(runningXT, runningYT, c='g', marker='+');
stairst = plt.scatter(stairsXT, stairsYT, c='b', marker='+');
walkingt = plt.scatter(walkingXT, walkingYT, c='y', marker='+');

plt.legend((car, internet, running, stairs, walking, cart, internett, runningt, stairst, walkingt), ("Car training data", "Internet training data", "Running training data", "Stairs training data", "Walking training data", "Car test data", "Internet test data", "Running test data", "Stairs test data", "Walking test data"), fontsize = 'x-small');

plt.show();

from sklearn import svm

svc = svm.SVC(kernel='linear');
svc.fit(trainSetX, trainSetY);
svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False);

print(svc.predict(testSetX));