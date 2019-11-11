import sklearn.metrics as met
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import theano
import theano.gpuarray
import pygpu
from pygpu import gpuarray
theano.gpuarray.use("cuda"+str(0))
from six.moves import cPickle
import os
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics as stats
import glob

f=open('obj.save','rb')
net=cPickle.load(f)
f.close()

y=np.load("cnn_same_config/y_test.npy")
preds=np.load("cnn_same_config/preds.npy")

y=y.reshape(y.shape[0])
preds=preds.reshape(preds.shape[0])

f1=met.f1_score(y, preds, average=None)
f1_w=met.f1_score(y, preds, average='weighted')
mcc=met.matthews_corrcoef(y,preds)

print('F1 Score: '+str(f1))
print('F1 Score Weighted: '+str(f1_w))
print("MCC: "+str(mcc))

np.save("f1",f1)

