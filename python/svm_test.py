import numpy as np 
from my_svm import SVM
import matplotlib.pyplot as plt 
# from svm import *
# from svmutil import *

'''y, x = svm_read_problem('data/a8a.txt')
prob  = svm_problem(y, x)
param = svm_parameter('-t 0 -c 1 -b 1')
m = svm_train(prob, param)'''
svm = SVM(training_path='data/a8a.txt', testing_path='data/a8a.t')

plt.figure(1)

svm.train(lr=0.001, iters=1000, C=100, batch_size=1)
x = range(1, len(svm.accuracy)+1)
plt.plot(x, svm.accuracy, 'b-',label="bs=1, "+str(max(svm.accuracy)))

svm.accuracy = []
svm.train(lr=0.001, iters=1000, C=100, batch_size=16)
plt.plot(x, svm.accuracy, 'r-',label="bs=16, "+str(max(svm.accuracy)))

svm.accuracy = []
svm.train(lr=0.001, iters=1000, C=100, batch_size=64)
plt.plot(x, svm.accuracy, 'g-',label="bs=64, "+str(max(svm.accuracy)))

svm.accuracy = []
svm.train(lr=0.001, iters=1000, C=100, batch_size=256)
plt.plot(x, svm.accuracy, 'm-',label="bs=256, "+str(max(svm.accuracy)))

svm.accuracy = []
svm.train(lr=0.00005, iters=1000, C=1, batch_size=None)
plt.plot(x, svm.accuracy, 'k-',label="GD, "+str(max(svm.accuracy)))

plt.legend(loc=0)
plt.grid()
plt.show()