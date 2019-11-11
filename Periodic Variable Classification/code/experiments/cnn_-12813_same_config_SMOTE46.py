import matplotlib
matplotlib.use('Agg')

import os
import copy
import generate
import numpy
import theano
import theano.gpuarray
import pygpu
from pygpu import gpuarray
#gpuarray.use("gpu"+str(0))
#import theano.sandbox.cuda
#theano.sandbox.cuda.use("gpu"+str(0))
#theano.gpuarray.use("gpu" + str(0))
theano.gpuarray.use("cuda" + str(0))
import lasagne
from nolearn.lasagne import NeuralNet, objective, TrainSplit, visualize
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import DenseLayer
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from util import plot_confusion, plot_misclassifications
import numpy as np
from six.moves import cPickle
import pickle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler 

# parameters
epochs = 500
learning_rate = 0.0002
verbose = 1
seed = 0
classes = [4,5,6]
#what are classes
	
test_size = 0.2

# get data and encode labels
X_2d, X_features, y, indices = generate.get_data("all", classes=classes, shuffle=True, seed=seed)
#print(scenario)
labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(y)
y = labelencoder.transform(y).astype(numpy.int32)
print("Total number of instances: " + str(len(y)))

# split data (train/test)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_2d, y, indices, test_size=test_size, random_state=seed)
X_test_plot = copy.deepcopy(X_test)
# why reshaping ?

sm=SMOTE(random_state=seed, ratio={0:3500,2:4000})
(f,g,h)=X_train.shape
X_train,y_train=sm.fit_sample(X_train.reshape(f,g*h),y_train)
X_train=X_train.reshape((X_train.shape[0],g,h))

y_train=y_train.astype(numpy.int32)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
print("Number of training instances: %i" % len(y_train))
print("Number of test instances: %i" % len(y_test))

layers = [
    (InputLayer, {'name':'input', 'shape': (None, X_train.shape[1], X_train.shape[2], X_train.shape[3])}),

    (Conv2DLayer, {'name':'conv2d1', 'num_filters': 16, 'filter_size': (5, 5), 'pad': 0, 'nonlinearity':rectify}),
    (MaxPool2DLayer, {'name':'maxpool1','pool_size': (2, 2)}),
    #(DropoutLayer, {'name':'dropout1','p':0.1}),

    (Conv2DLayer, {'name':'conv2d2','num_filters': 64, 'filter_size': (5, 5), 'pad': 0, 'nonlinearity':rectify}),
    #(MaxPool2DLayer, {'pool_size': (2, 2)}),
    #(DropoutLayer, {'p':0.3}),         

    #(Conv2DLayer, {'name':'conv2d3','num_filters': 256, 'filter_size': (5, 5), 'pad': 0, 'nonlinearity':rectify}),
    #(MaxPool2DLayer, {'pool_size': (2, 2)}),
    #(DropoutLayer, {'p':0.5}),       

    #(DenseLayer, {'name':'dense1','num_units': 512}),
    #(DropoutLayer, {'name':'dropout2','p':0.5}),
    #(DenseLayer, {'name':'dense2','num_units': 512}),

    (DenseLayer, {'name':'output','num_units': len(list(set(y))), 'nonlinearity': softmax}),
]

net = NeuralNet(
    layers=layers,
    #layers=[('input', InputLayer),
    #        ('conv2d1', Conv2DLayer),
    #        ('maxpool1', MaxPool2DLayer),
    #        ('dropout1', DropoutLayer),
    #        ('conv2d2', Conv2DLayer),
    #	    ('conv2d3', Conv2DLayer),
    #        ('dense1', DenseLayer),
    #        ('dropout2', DropoutLayer),
    #	    ('dense2', DenseLayer),
    #        ('output', DenseLayer),
    #        ],
    max_epochs=epochs,
    update=lasagne.updates.adam,
    update_learning_rate=learning_rate,
    objective_l2=0.0025,
    train_split=TrainSplit(eval_size=0.05),
    verbose=verbose,
)

net.fit(X_train, y_train)
preds = net.predict(X_test)
preds_proba = net.predict_proba(X_test)
acc = accuracy_score(y_test, preds)
print("Accuracy: %f" % acc)

y_test = labelencoder.inverse_transform(y_test)
preds = labelencoder.inverse_transform(preds)
y_train = labelencoder.inverse_transform(y_train)

# plot misclassifications
plot_misclassifications(y_test, preds, X_test_plot, indices_test, "cnn/misclassifications")

# save output
numpy.save("cnn/X_test",X_test)
numpy.save("cnn/y_test",y_test)
numpy.save("cnn/X_train",X_train)
numpy.save("cnn/y_train",y_train)
numpy.save("cnn/preds_proba",preds_proba)
numpy.save("cnn/preds",preds)
numpy.savetxt("cnn/y_test_cnn.csv", y_test, delimiter=",", fmt='%.4f')
numpy.savetxt("cnn/preds_cnn.csv", preds, delimiter=",", fmt='%.4f')
numpy.savetxt("cnn/preds_proba_cnn.csv", preds_proba, delimiter=",", fmt='%.4f')
plot_confusion(y_test, preds, "cnn/confusion_cnn_hpercent.png")
#plt1=visualize.plot_conv_weights(net.layers_['conv2d1'])
#plt1.savefig("cnn/filters1.png")
#plt1.close()
#plt2=visualize.plot_conv_weights(net.layers_['conv2d2'])
#plt2.savefig("cnn/filters2.png")
#plt3=visualize.plot_conv_weights(net.layers_['conv2d3'])
#plt3.savefig("cnn/filters3.png")
#plt3.close()

f=open('obj.save','wb')
cPickle.dump(net,f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

print("F1 Score: "+str(f1_score(y_test.reshape(y_test.shape[0]),preds.reshape(preds.shape[0]),average=None)))
print("Matthews correlation coefficient (MCC): "+str(matthews_corrcoef(y_test.reshape(y_test.shape[0]),preds.reshape(preds.shape[0]))))

#f=open('obj.save','rb')
#net=cPickle.load(f)
#f.close()

#print(net)

