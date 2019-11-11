import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.7, N+4)
    return mycmap

mycmap = transparent_cmap(plt.cm.Greens)

x=np.load("../X_test.npy")
y=np.load("../y_test.npy")
m=np.load("m_a.npy")
ww=np.load("w.npy")
tt=np.load("t.npy")

os.mkdir("heatmaps")
os.mkdir("heatmaps/1")
os.mkdir("heatmaps/2")
os.mkdir("heatmaps/4")
os.mkdir("heatmaps/5")
os.mkdir("heatmaps/6")
os.mkdir("heatmaps/8")
os.mkdir("heatmaps/13")

(wh,h,w)=x[99].shape
u,v=np.mgrid[0:h,0:w]
for i in range(len(tt[0])):
    fig,ax=plt.subplots(1,1)
    ax.imshow(x[ww[0][tt[0][i]]].reshape((23,24)))
    cb=ax.contourf(v,u,m[ww[0][tt[0][i]]],15,cmap=mycmap)
    plt.colorbar(cb)
    plt.savefig("heatmaps/"+str(y[ww[0][tt[0][i]]])+"/"+str(ww[0][tt[0][i]])+"_"+str(y[ww[0][tt[0][i]]])+".png")
    plt.close()
