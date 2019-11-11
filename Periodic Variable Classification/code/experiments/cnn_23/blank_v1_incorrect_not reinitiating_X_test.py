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
#import multiprocessing

classes=[1,2,4,5,6,8,13]
bins=10
os.mkdir("blank_v1")

x=np.load("X_test.npy")
(l,o,m,n)=x.shape
y=np.load("y_test.npy")
p=np.load("preds_proba.npy")
pr=np.load("preds.npy")

# to load the trained cnn which is to be used for prediction
f=open('../obj.save','rb') 
net=cPickle.load(f)
f.close()

m_a=np.zeros((l,m,n))

# dictionary to be used as indices for various prediction and prediction probabilities numpy array 
c={1:0,2:1,4:2,5:3,6:4,8:5,13:6}

# w numpy array stores indices of objects in x (a 4D numpy array storing all the test dmdts)
# such that the prediction == test label 
w=np.where(y==pr)

# t numpy array stores indices of objects in w such that the prediction probability of the predicted class
# is greater than or equal to 0.8
t=np.where(p[w[0]]>=0.8)
for i in range(len(t[0])):
    for j in range(m):
        for k in range(n):
            if x[w[0][t[0][i]],0,j,k]!=0.0:
                x[w[0][t[0][i]],0,j,k]=0.0
                y_test=net.predict_proba(x[w[0][t[0][i]]].reshape((1,1,23,24)))
                m_a[w[0][t[0][i]],j,k]=y_test[0,c[y[w[0][t[0][i]]]]]-p[0,c[y[w[0][t[0][i]]]]]
    print(i)
##n_jobs=2
##pool=multiprocessing.Pool(n_jobs)
##results=pool.map(mp,range(l))

np.save("blank_v1/m_a",m_a)
np.save("blank_v1/w",w)
np.save("blank_v1/t",t)

##np.save("blank/m_a",m_a)
## 
##
##h_1=[]
##h_2=[]
##h_4=[]
##h_5=[]
##h_6=[]
##h_8=[]
##h_13=[]
##
##for i in range(l):
##    print(float(i))
##    if y[i]==1:
##        if y[i]==pr[i]:
##            c=1
##            if p[i,c-1]>=0.50 and p[i,c-1]<=1.00:
##                h_1=h_1+[p[i,c-1]]
##    if y[i]==2:
##        if y[i]==pr[i]:
##            c=2
##            if p[i,c-1]>=0.50 and p[i,c-1]<=1.00:
##                h_2=h_2+[p[i,c-1]]
##    if y[i]==6:
##        if y[i]==pr[i]:
##            c=5
##            if p[i,c-1]>=0.50 and p[i,c-1]<=1.00:
##                h_6=h_6+[p[i,c-1]]
##    if y[i]==4:
##        if y[i]==pr[i]:
##            c=3
##            if p[i,c-1]>=0.50 and p[i,c-1]<=1.00:
##                h_4=h_4+[p[i,c-1]]
##    if y[i]==5:
##        if y[i]==pr[i]:
##            c=4
##            if p[i,c-1]>=0.50 and p[i,c-1]<=1.00:
##                h_5=h_5+[p[i,c-1]]
##    if y[i]==8:
##        if y[i]==pr[i]:
##            c=6
##            if p[i,c-1]>=0.50 and p[i,c-1]<=1.00:
##                h_8=h_8+[p[i,c-1]]
##    if y[i]==13:
##        if y[i]==pr[i]:
##            c=7
##            if p[i,c-1]>=0.50 and p[i,c-1]<=1.00:
##                h_13=h_13+[p[i,c-1]]
##                
##		
##
##np.save("blank/h_1",h_1)
##np.save("blank/h_2",h_2)
##np.save("blank/h_6",h_6)
##np.save("blank/h_4",h_4)
##np.save("blank/h_5",h_5)
##np.save("blank/h_8",h_8)
##np.save("blank/h_13",h_13)
##
##    
##plt.hist(h_1,bins,facecolor='blue',alpha=0.5)
##plt.title("EW")
##plt.xlabel("Probabilities")
##plt.ylabel("Number of Objects")
##plt.savefig("blank/class1.png")
##plt.close()
##
##plt.hist(h_2,bins,facecolor='blue',alpha=0.5)
##plt.title("EA")
##plt.xlabel("Probabilities")
##plt.ylabel("Number of Objects")
##plt.savefig("blank/class2.png")
##plt.close()
##
##plt.hist(h_6,bins,facecolor='blue',alpha=0.5)
##plt.title("RRd")
##plt.xlabel("Probabilities")
##plt.ylabel("Number of Objects")
##plt.savefig("blank/class6.png")
##plt.close()
##
##plt.hist(h_4,bins,facecolor='blue',alpha=0.5)
##plt.title("RRab")
##plt.xlabel("Probabilities")
##plt.ylabel("Number of Objects")
##plt.savefig("blank/class4.png")
##plt.close()
##
##plt.hist(h_5,bins,facecolor='blue',alpha=0.5)
##plt.title("RRc")
##plt.xlabel("Probabilities")
##plt.ylabel("Number of Objects")
##plt.savefig("blank/class5.png")
##plt.close()
##
##plt.hist(h_8,bins,facecolor='blue',alpha=0.5)
##plt.title("RSCVn")
##plt.xlabel("Probabilities")
##plt.ylabel("Number of Objects")
##plt.savefig("blank/class8.png")
##plt.close()
##
##plt.hist(h_13,bins,facecolor='blue',alpha=0.5)
##plt.title("LPV")
##plt.xlabel("Probabilities")
##plt.ylabel("Number of Objects")
##plt.savefig("blank/class13.png")
##plt.close()

##############################################################################

##    if y[i]==1:
##        c=1
##    if y[i]==2:
##        c=2
##    if y[i]==4:
##        c=3
##    if y[i]==5:
##        c=4
##    if y[i]==6:
##        c=5
##    if y[i]==8:
##        c=6
##    if y[i]==13:
##        c=7

##    if p[i,c-1]<0.90:
##        for j in range(m):
##            for k in range(n):
##                m_a[i,j,k]=9
##                
##    if y[i]!=pr[i]:
##        for j in range(m):
##            for k in range(n):
##                m_a[i,j,k]=None




    
                
    
    
