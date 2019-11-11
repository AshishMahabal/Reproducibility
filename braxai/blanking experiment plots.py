import matplotlib.pyplot as plt
import numpy as np
import os

x=np.load("x_test.npy")
(l,m,n,o)=x.shape
y=np.load("y_test.npy")
p=np.load("preds_proba.npy")
pr=np.load("preds.npy")
m_a=np.load("m_a.npy")

clss='bogus'
#os.mkdir("blanking")
#os.mkdir("keras-vis/bogus/")
os.mkdir("keras-vis/bogus/blanking/")
#os.mkdir("blanking/"+clss+"/dmdt/")
#os.mkdir("blanking/"+clss+"/blank/")

#%matplotlib inline
for i in range(x.shape[0]):
  if y[i]==pr[i] and max(p[i])>=0.8 and pr[i]==0:
    #plt.figure()
    #plt.axis("off")
    #plt.imshow(x[i][:,:,0])
    #plt.savefig("blanking/"+clss+"/dmdt/"+str(i)+".png", bbox_inches='tight', transparent="True", pad_inches=0)
    #plt.close()
    plt.figure()
    plt.imshow(m_a[i])
    plt.axis("off")
    plt.savefig("keras-vis/bogus/blanking/"+str(i)+".png", bbox_inches='tight', transparent="True", pad_inches=0)
    plt.close()
