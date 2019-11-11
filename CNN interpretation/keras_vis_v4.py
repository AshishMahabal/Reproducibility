
# coding: utf-8

# In[1]:


import configparser
import pandas as pd
import theano
import theano.gpuarray
theano.gpuarray.use("cuda" + str(0))
from nolearn.lasagne import visualize
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pygpu
from pygpu import gpuarray
from six.moves import cPickle
import os
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics as stats
import glob
from texttable import Texttable
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

#import matplotlib
#matplotlib.use('Agg')

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#from utils.util import plot_confusion
#import utils.generate as gen

import copy
#import generate
import numpy as np
from sklearn import metrics
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn
import keras
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.engine.topology import Input
from keras.optimizers import Adam
from keras import regularizers
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import lasagne

from vis.visualization import visualize_activation, visualize_saliency, visualize_cam
from vis.utils import utils
from keras import activations


f=open('periodic/p_obj.save','rb')
net=cPickle.load(f)
f.close()


# In[2]:


model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", 
                 #data_format = "channels_first",
                 kernel_regularizer=regularizers.l2(0.01), 
                 input_shape=(22, 24, 1),padding='valid',
                 name="conv2d1"))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', 
                  #     data_format = "channels_first",
                       name="maxpool2d1"
                      ))
#model.add(Dropout(rate=0.1))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu", 
                 kernel_regularizer=regularizers.l2(0.01),
                 padding='valid', 
                 #data_format = "channels_first",
                 name="conv2d2"
                ))

#model.add(Conv2D(filters=256, kernel_size=(5, 5), activation="relu", kernel_regularizer=regularizers.l2(0.01)))

model.add(Flatten(
    #data_format = "channels_first", 
    name="flatten"))
#model.add(Dense(units=512, kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(rate=0.5))
#model.add(Dense(units=512, kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(units=7, activation="softmax", name="preds"))


# In[3]:


weights = lasagne.layers.get_all_param_values(net.get_all_layers()[-1])


# In[4]:


#weights[0]=weights[0].reshape((5,5,1,16))
#weights[2]=weights[2].reshape((5,5,16,64))
weights[0]=np.transpose(weights[0],(2,3,1,0))
weights[2]=np.transpose(weights[2],(2,3,1,0))


# In[5]:


#weights[0].shape


# In[6]:


#weights[0].reshape((5,5,1,16))
model.set_weights(weights)


# In[7]:


layer_idx = utils.find_layer_idx(model, 'preds')
# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)


# In[58]:


#x_train=np.load("periodic/X_train.npy")
#x_test=np.load("periodic/X_test.npy")
x=np.load("periodic/X_test.npy")

#y_train=np.load("periodic/y_train.npy")
#y_test=np.load("periodic/y_test.npy")
y=np.load("periodic/y_test.npy")

#x=np.concatenate((x_train, x_test))
#y=np.concatenate((y_train, y_test))

preds=np.load("periodic/preds.npy")
prob=np.load("periodic/preds_proba.npy")

classes=[1,2,4,5,6,8,13]
filterid={1:0,2:1,4:2,5:3,6:4,8:5,13:6}
pd={1:'EW',2:'EA',4:'RRab',5:'RRc',6:'RRd',8:'RSCVn',13:'LPV'}

dmints = [-8,-5,-3,-2.5,-2,-1.5,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,5,8]
dtints = [0,1.0/145,2.0/145,3.0/145,4.0/145,1.0/25,2.0/25,3.0/25,1.5,2.5,3.5,4.5,5.5,7,10,20,30,60,90,120,240,600,960,2000,4000]

xloc=np.arange(25)
yloc=np.arange(23)
yloc=yloc[::-1]
for i in range(len(dtints)):
    dtints[i]=round(dtints[i],3)
yloc=yloc-0.5
xloc=xloc-0.5

thres=np.where(prob>=0.8)
ind_thres=thres[0]
class_thres=thres[1]

#os.mkdir("keras-vis")
os.mkdir("keras-vis/activation_maximization/")
os.mkdir("keras-vis/activation_maximization/preds")


# In[57]:


#from matplotlib import pyplot as plt
##%matplotlib inline
#plt.rcParams['figure.figsize'] = (18, 6)
#
#layer_idx = utils.find_layer_idx(model, 'preds')
## Swap softmax with linear
##model.layers[layer_idx].activation = activations.linear
##model = utils.apply_modifications(model)
#
## This is the output node we want to maximize.
#filter_idx = 1
#input_range= (0.,1.)
#img_act = visualize_activation(model, 
#                               layer_idx, 
#                               filter_indices=filter_idx, 
#                               input_range=input_range,
#                               #verbose=True,
#                               tv_weight=0.2, 
#                               lp_norm_weight=0.
#                              )
#fig, ax=plt.subplots(1,1)
#im=ax.imshow(img_act[:,:,0])
#fig.colorbar(im)
#ax.set_xticks(xloc)
#ax.set_xticklabels(dtints,rotation=90)
#ax.set_yticks(yloc)
#ax.set_yticklabels(dmints)
#plt.title("Filter for Class EA")
#plt.savefig("keras-vis/activation_maximization/preds/EA.png")
#plt.close()


# In[59]:


#model.layers[-1].activation


# In[25]:


#img_act.shape


# In[60]:


#plt.rcParams['figure.figsize'] = (18, 6)
#layer_idx = utils.find_layer_idx(model, 'preds')
#penultimate_layer_idx = utils.find_layer_idx(model, 'conv2d2')
## Swap softmax with linear
##model.layers[layer_idx].activation = activations.linear
##model = utils.apply_modifications(model)
#os.mkdir("keras-vis/grad_CAM/")
#for i in range(len(classes)):
#    # This is the output node we want to maximize.
#    filter_idx = filterid[classes[i]]
#    os.mkdir("keras-vis/grad_CAM/"+pd[classes[i]]+"/")
#    #input_range= (0,255)
#    #image=np.random.random_sample((22,24,1))
#    x_ind=np.where(class_thres==filter_idx)[0]
#    for ii in range(x_ind.shape[0]):
#        image=x[ind_thres[x_ind[ii]]].transpose((1,2,0))
#        img_cam = visualize_cam(model, layer_idx, filter_indices=filter_idx, seed_input=image, 
#                                     penultimate_layer_idx=penultimate_layer_idx)
#        plt.figure()
#        fig, ax=plt.subplots(1,2)
#        im1=ax[0].imshow(image[:,:,0])
#        im2=ax[1].imshow(img_cam)
#        divider1 = make_axes_locatable(ax[0])
#        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
#        divider2 = make_axes_locatable(ax[1])
#        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
#        ax[0].set_xticks(xloc)
#        ax[0].set_xticklabels(dtints,rotation=90)
#        ax[1].set_xticks(xloc)
#        ax[1].set_xticklabels(dtints,rotation=90)
#        ax[0].set_yticks(yloc)
#        ax[0].set_yticklabels(dmints)
#        ax[1].set_yticks(yloc)
#        ax[1].set_yticklabels(dmints)
#        ax[0].set(xlabel="dt(days)",ylabel="dm(mag)")
#        ax[1].set(xlabel="dt(days)",ylabel="dm(mag)")
#        fig.colorbar(im1, cax=cax1)
#        fig.colorbar(im2, cax=cax2)
#        plt.tight_layout()
#        plt.suptitle("Class: "+pd[classes[i]]+", grad_CAM,\n"+"X_id: "+str(ind_thres[x_ind[ii]])+"\nPred_Prob: "+str(round(prob[ind_thres[x_ind[ii]],class_thres[x_ind[ii]]],4)))
#        plt.savefig("keras-vis/grad_CAM/"+pd[classes[i]]+"/"+str(ind_thres[x_ind[ii]])+".png")
#        plt.close()


# In[27]:


plt.rcParams['figure.figsize'] = (18, 6)
layer_idx = utils.find_layer_idx(model, 'preds')
penultimate_layer_idx = utils.find_layer_idx(model, 'conv2d2')
# Swap softmax with linear
#model.layers[layer_idx].activation = activations.linear
#model = utils.apply_modifications(model)
os.mkdir("keras-vis/saliency/")
classes=[2,4,5,6,8,13,1]
for i in range(len(classes)):
    # This is the output node we want to maximize.
    filter_idx = filterid[classes[i]]
    os.mkdir("keras-vis/saliency/"+pd[classes[i]]+"/")
    #input_range= (0,255)
    #image=np.random.random_sample((22,24,1))
    x_ind=np.where(class_thres==filter_idx)[0]
    for ii in range(x_ind.shape[0]):
        image=x[ind_thres[x_ind[ii]]].transpose((1,2,0))
        img_sal = visualize_saliency(model, layer_idx, filter_indices=filter_idx, seed_input=image, 
                             #penultimate_layer_idx=penultimate_layer_idx
                            )
        plt.figure()
        fig, ax=plt.subplots(1,2)
        im1=ax[0].imshow(image[:,:,0])
        im2=ax[1].imshow(img_sal)
        divider1 = make_axes_locatable(ax[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        divider2 = make_axes_locatable(ax[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        ax[0].set_xticks(xloc)
        ax[0].set_xticklabels(dtints,rotation=90)
        ax[1].set_xticks(xloc)
        ax[1].set_xticklabels(dtints,rotation=90)
        ax[0].set_yticks(yloc)
        ax[0].set_yticklabels(dmints)
        ax[1].set_yticks(yloc)
        ax[1].set_yticklabels(dmints)
        ax[0].set(xlabel="dt(days)",ylabel="dm(mag)")
        ax[1].set(xlabel="dt(days)",ylabel="dm(mag)")
        fig.colorbar(im1, cax=cax1)
        fig.colorbar(im2, cax=cax2)
        plt.tight_layout()
        plt.suptitle("Class: "+pd[classes[i]]+", Saliency,\n"+"X_id: "+str(ind_thres[x_ind[ii]])+"\nPred_Prob: "+str(round(prob[ind_thres[x_ind[ii]],class_thres[x_ind[ii]]],4)))
        plt.savefig("keras-vis/saliency/"+pd[classes[i]]+"/"+str(ind_thres[x_ind[ii]])+".png")
        plt.close()

