
# coding: utf-8

# In[3]:


import numpy as np
import random
import matplotlib.pyplot as plt
import glob

dat=glob.glob("indiv_dmdt/*npy")
#x=np.load("X_2d.npy")
#(l,m,n)=x.shape
l=len(dat)
(m,n)=np.load(dat[0]).shape

dmints = [-8,-5,-3,-2.5,-2,-1.5,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,5,8]
dtints = [0.0,1.0/145,2.0/145,3.0/145,4.0/145,1.0/25,2.0/25,3.0/25,1.5,2.5,3.5,4.5,5.5,7,10,20,30,60,90,120,240,600,960,2000,4000]
xloc=np.arange(m+1)-0.5
b=int(0.001*l)


# In[4]:


for i in range(b):
    r=random.randrange(l)
    x=np.load(dat[r])
    h=np.zeros(m)
    for j in range(m):
        for k in range(n):
            h[j]=h[j]+x[j,k]
            
    fig,ax=plt.subplots(1,1)
    plt.bar(np.arange(m),h)
    plt.title(r)
    plt.xlabel("dm bins")
    plt.ylabel("Number of Objects")
    loc, labels=plt.xticks()
    ax.set_xticks(xloc)
    ax.set_xticklabels(dmints,rotation=90)
    plt.savefig("hist_dm/"+str(r)+".png")
    plt.close()
    #print(loc)
    #print(labels)
    
#print(r,h)

