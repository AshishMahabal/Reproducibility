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


classes=[1,2,4,5,6,8,13]
bins=10

f=open('../obj.save','rb')
net=cPickle.load(f)
f.close()

p_1=np.zeros((23,24))
p_2=np.zeros((23,24))
p_4=np.zeros((23,24))
p_5=np.zeros((23,24))
p_6=np.zeros((23,24))
p_8=np.zeros((23,24))
p_13=np.zeros((23,24))

for j in range(23):
    for k in range(24):
        x=np.zeros((23,24))
        x[j,k]=255.0
        [p_1[j,k],p_2[j,k],p_4[j,k],p_5[j,k],p_6[j,k],p_8[j,k],p_13[j,k]]=net.predict_proba(x.reshape((1,1,23,24)))[0]

os.mkdir("lighten")
np.save("lighten/p_1",p_1)
np.save("lighten/p_2",p_2)
np.save("lighten/p_4",p_4)
np.save("lighten/p_5",p_5)
np.save("lighten/p_6",p_6)
np.save("lighten/p_8",p_8)
np.save("lighten/p_13",p_13)
np.save("lighten/x",x)

# defining figure grid
h=23
w=24
u,v=np.mgrid[0:h,0:w]

dmints = [-8,-5,-3,-2.5,-2,-1.5,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,5,8]
dtints = [1.0/145,2.0/145,3.0/145,4.0/145,1.0/25,2.0/25,3.0/25,1.5,2.5,3.5,4.5,5.5,7,10,20,30,60,90,120,240,600,960,2000,4000]

xloc=np.arange(24)
yloc=np.arange(23)
yloc=yloc[::-1]
for i in range(len(dtints)):
    dtints[i]=round(dtints[i],3)
    
xloc=xloc[1::2]
yloc=yloc[0::2]
dmints=dmints[0::2]
dtints=dtints[1::2]

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_1)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
ax.set_title("Activation Probabilities for class1: EW"+"\nMin: "+str(round(p_1.min(),5))+" Violet"+"\n Max: "+str(round(p_1.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/p_1.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_2)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
ax.set_title("Activation Probabilities for class2: EA"+"\nMin: "+str(round(p_2.min(),5))+" Violet"+"\n Max: "+str(round(p_2.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/p_2.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_4)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
ax.set_title("Activation Probabilities for class4: RRab"+"\nMin: "+str(round(p_4.min(),5))+" Violet"+"\n Max: "+str(round(p_4.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/p_4.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_5)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
ax.set_title("Activation Probabilities for class5: RRc"+"\nMin: "+str(round(p_5.min(),5))+" Violet"+"\n Max: "+str(round(p_5.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/p_5.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_6)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
ax.set_title("Activation Probabilities for class6: RRd"+"\nMin: "+str(round(p_6.min(),5))+" Violet"+"\n Max: "+str(round(p_6.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/p_6.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_8)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
ax.set_title("Activation Probabilities for class8: RSCVn"+"\nMin: "+str(round(p_8.min(),5))+" Violet"+"\n Max: "+str(round(p_8.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/p_8.png")
plt.close()

fig,ax=plt.subplots(1,1)
im1=ax.imshow(p_13)
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.xticks(xloc,dtints,rotation=90)
plt.yticks(yloc,dmints)
ax.set_xticks(xloc)
ax.set_xticklabels(dtints,rotation=90)
ax.set_yticks(yloc)
ax.set_yticklabels(dmints)
ax.set(xlabel="dt(days)",ylabel="dm(mag)")
ax.set_title("Activation Probabilities for class13: LPV"+"\nMin: "+str(round(p_13.min(),5))+" Violet"+"\n Max: "+str(round(p_13.max(),5))+" Yellow")
fig.colorbar(im1,cax=cax1,boundaries=np.linspace(0,1,10))
plt.tight_layout()
#plt.ticks.set_xspacing(0.0005*mul)
plt.savefig("lighten/p_13.png")
plt.close()
