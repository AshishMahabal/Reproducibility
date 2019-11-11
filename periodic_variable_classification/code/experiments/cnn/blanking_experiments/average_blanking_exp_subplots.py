import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# dictionary useful in indexing
# of loaded numpy arrays as each
# column in each of these numpy
# array represents a class

c={1:0,2:1,4:2,5:3,6:4,8:5,13:6} 

# creating a directory for storing the averaged results
os.mkdir("avg_sub")

# loading w numpy array which contains indices of objects in
# y(prediction through cnn) which are
# classified correctly
w=np.load("w.npy")

# loading t numpy array which contains indices of objects in w which
# have prediction probability greater than or equal to 0.8
t=np.load("t.npy")

# loading the dmdts
x=np.load("../X_test.npy")

# one numpy array contains indices of objects in x numpy array which
# belong to class1
# Similar for the rest of the classes
one=np.where(t[1]==c[1])
two=np.where(t[1]==c[2])
four=np.where(t[1]==c[4])
five=np.where(t[1]==c[5])
six=np.where(t[1]==c[6])
eight=np.where(t[1]==c[8])
thirteen=np.where(t[1]==c[13])

(l,o,m,n)=x.shape

# creating empty lists 
x_1=[]
x_2=[]
x_4=[]
x_5=[]
x_6=[]
x_8=[]
x_13=[]

# concatenating the objects belonging to a particular class
# into a list for each of the classes
for i in range(len(one[0])):
    x_1=x_1+[x[w[0][t[0][one[0][i]]]]]
for i in range(len(two[0])):
    x_2=x_2+[x[w[0][t[0][two[0][i]]]]]
for i in range(len(four[0])):
    x_4=x_4+[x[w[0][t[0][four[0][i]]]]]
for i in range(len(five[0])):
    x_5=x_5+[x[w[0][t[0][five[0][i]]]]]
for i in range(len(six[0])):
    x_6=x_6+[x[w[0][t[0][six[0][i]]]]]
for i in range(len(eight[0])):
    x_8=x_8+[x[w[0][t[0][eight[0][i]]]]]
for i in range(len(thirteen[0])):
    x_13=x_13+[x[w[0][t[0][thirteen[0][i]]]]]

# converting the concatenated lists into a numpy array
x_1=np.array(x_1)
x_2=np.array(x_2)
x_4=np.array(x_4)
x_5=np.array(x_5)
x_6=np.array(x_6)
x_8=np.array(x_8)
x_13=np.array(x_13)

# saving the numpy arrays
np.save("avg_sub/x_1",x_1)
np.save("avg_sub/x_2",x_2)
np.save("avg_sub/x_4",x_4)
np.save("avg_sub/x_5",x_5)
np.save("avg_sub/x_6",x_6)
np.save("avg_sub/x_8",x_8)
np.save("avg_sub/x_13",x_13)

# computing average dmdt for each of the classes
if x_1.size!=0: # checking whether there exists a class which has no object being classified by cnn
    one_sum=np.zeros((23,24))
    for j in range(x_1.shape[2]):
        for k in range(x_1.shape[3]):
            for i in range(x_1.shape[0]):
                one_sum[j,k]=one_sum[j,k]+x_1[i,0,j,k]
    one_avg=one_sum/x_1.shape[0]
    np.save("avg_sub/1_mean",one_avg)
    
if x_2.size!=0:
    two_sum=np.zeros((23,24))
    for j in range(x_2.shape[2]):
        for k in range(x_2.shape[3]):
            for i in range(x_2.shape[0]):
                two_sum[j,k]=two_sum[j,k]+x_2[i,0,j,k]
    two_avg=two_sum/x_2.shape[0]
    np.save("avg_sub/2_mean",two_avg)

if x_4.size!=0:
    four_sum=np.zeros((23,24))
    for j in range(x_4.shape[2]):
        for k in range(x_4.shape[3]):
            for i in range(x_4.shape[0]):
                four_sum[j,k]=four_sum[j,k]+x_4[i,0,j,k]
    four_avg=four_sum/x_4.shape[0]
    np.save("avg_sub/4_mean",four_avg)

if x_5.size!=0:
    five_sum=np.zeros((23,24))
    for j in range(x_5.shape[2]):
        for k in range(x_5.shape[3]):
            for i in range(x_5.shape[0]):
                five_sum[j,k]=five_sum[j,k]+x_5[i,0,j,k]
    five_avg=five_sum/x_5.shape[0]
    np.save("avg_sub/5_mean",five_avg)

if x_6.size!=0:
    six_sum=np.zeros((23,24))
    for j in range(x_6.shape[2]):
        for k in range(x_6.shape[3]):
            for i in range(x_6.shape[0]):
                six_sum[j,k]=six_sum[j,k]+x_6[i,0,j,k]
    six_avg=six_sum/x_6.shape[0]
    np.save("avg_sub/6_mean",six_avg)

if x_8.size!=0:
    eight_sum=np.zeros((23,24))
    for j in range(x_8.shape[2]):
        for k in range(x_8.shape[3]):
            for i in range(x_8.shape[0]):
                eight_sum[j,k]=eight_sum[j,k]+x_8[i,0,j,k]
    eight_avg=eight_sum/x_8.shape[0]
    np.save("avg_sub/8_mean",eight_avg)

if x_13.size!=0:
    thirteen_sum=np.zeros((23,24))
    for j in range(x_13.shape[2]):
        for k in range(x_13.shape[3]):
            for i in range(x_13.shape[0]):
                thirteen_sum[j,k]=thirteen_sum[j,k]+x_13[i,0,j,k]
    thirteen_avg=thirteen_sum/x_13.shape[0]
    np.save("avg_sub/13_mean",thirteen_avg)

# loading m_a numpy array which contains information regarding
# the changes in probabilities encountered by blanking out individual pixels
m=np.load("m_a.npy")

# creating numpy arrays for getting the average change in probabilities for
# each of the classes
m_1=np.zeros((23,24))
m_2=np.zeros((23,24))
m_4=np.zeros((23,24))
m_5=np.zeros((23,24))
m_6=np.zeros((23,24))
m_8=np.zeros((23,24))
m_13=np.zeros((23,24))

# summing the changes in probabilities for each of the classes
for i in range(len(one[0])):
    m_1=m_1+m[w[0][t[0][one[0][i]]]]

for i in range(len(two[0])):
    m_2=m_2+m[w[0][t[0][two[0][i]]]]

for i in range(len(four[0])):
    m_4=m_4+m[w[0][t[0][four[0][i]]]]

for i in range(len(five[0])):
    m_5=m_5+m[w[0][t[0][five[0][i]]]]

for i in range(len(six[0])):
    m_6=m_6+m[w[0][t[0][six[0][i]]]]

for i in range(len(eight[0])):
    m_8=m_8+m[w[0][t[0][eight[0][i]]]]

for i in range(len(thirteen[0])):
    m_13=m_13+m[w[0][t[0][thirteen[0][i]]]]

# computing the average of changes in probabilities for each of the classes
if len(one[0])!=0:# checking whether there exists a class which has no object being classified by cnn
    m_1_avg=m_1/len(one[0])
    np.save("avg_sub/m_1_mean",m_1_avg)
if len(two[0])!=0:
    m_2_avg=m_2/len(two[0])
    np.save("avg_sub/m_2_mean",m_2_avg)
if len(four[0])!=0:
    m_4_avg=m_4/len(four[0])
    np.save("avg_sub/m_4_mean",m_4_avg)
if len(five[0])!=0:
    m_5_avg=m_5/len(five[0])
    np.save("avg_sub/m_5_mean",m_5_avg)
if len(six[0])!=0:
    m_6_avg=m_6/len(six[0])
    np.save("avg_sub/m_6_mean",m_6_avg)
if len(eight[0])!=0:
    m_8_avg=m_8/len(eight[0])
    np.save("avg_sub/m_8_mean",m_8_avg)
if len(thirteen[0])!=0:
    m_13_avg=m_13/len(thirteen[0])
    np.save("avg_sub/m_13_mean",m_13_avg)

# defining a function for translucent/transparent colormap
# to be used in heatmaps of averaged changes in probabilities
# being overlayed on the averaged dmdts for each of the classes

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.7, N+4)
    return mycmap

def inv_transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0.7, 0, N+4)
    return mycmap

# assigning the colormap for heatmaps
mycmap = transparent_cmap(plt.cm.Greens)
inv_mycmap = inv_transparent_cmap(plt.cm.Greens_r)

# defining figure grid
(wh,h,w)=x[99].shape
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
    
##dmdt bins with size according to its original scale 
##xloc=(23.0/4000)*np.array(dtints)
##for i in range(len(dtints)):
##    dtints[i]=round(dtints[i],3)
##yloc=(-1*(22.0/16)*np.array(dmints))+11

mul=2 #multiplier for xlabels spacing

# plotting the heatmaps representing the averaged changes in
# probabilities which are overlayed on the averaged
# dmdts for each of the classes
if len(one[0])!=0:
    fig,ax=plt.subplots(1,2)
    im1=ax[0].imshow(one_avg.reshape((23,24)))
    im2=ax[1].imshow(m_1_avg)
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    ##cb=ax.contourf(v,u,m_1_avg,15,cmap=inv_mycmap)
    ##plt.colorbar(cb)
    plt.xticks(xloc,dtints,rotation=90)
    plt.yticks(yloc,dmints)
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
    ax[0].set_title("Averaged dmdt for class1 ("+str(x_1.shape[0])+")"+"\nMin: "+str(round(one_avg.min(),5))+" Violet"+"\n Max: "+str(round(one_avg.max(),5))+" Yellow")
    ax[1].set_title("Averaged Change\n in Probabilities\n Min: "+str(round(m_1_avg.min(),5))+" Violet"+"\n Max: "+str(round(m_1_avg.max(),5))+" Yellow")
    fig.colorbar(im1,cax=cax1)
    fig.colorbar(im2,cax=cax2)
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("avg_sub/1_mean_heatmap.png")
    plt.close()

if len(two[0])!=0:
    fig,ax=plt.subplots(1,2)
    im1=ax[0].imshow(two_avg.reshape((23,24)))
    im2=ax[1].imshow(m_2_avg)
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    ##cb=ax.contourf(v,u,m_1_avg,15,cmap=inv_mycmap)
    ##plt.colorbar(cb)
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
    ax[0].set_title("Averaged dmdt for class2 ("+str(x_2.shape[0])+")"+"\nMin: "+str(round(two_avg.min(),5))+" Violet"+"\n Max: "+str(round(two_avg.max(),5))+" Yellow")
    ax[1].set_title("Averaged Change\n in Probabilities\n Min: "+str(round(m_2_avg.min(),5))+" Violet"+"\n Max: "+str(round(m_2_avg.max(),5))+" Yellow")
    fig.colorbar(im1,cax=cax1)
    fig.colorbar(im2,cax=cax2)
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("avg_sub/2_mean_heatmap.png")
    plt.close()

if len(four[0])!=0:
    fig,ax=plt.subplots(1,2)
    im1=ax[0].imshow(four_avg.reshape((23,24)))
    im2=ax[1].imshow(m_4_avg)
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    ##cb=ax.contourf(v,u,m_1_avg,15,cmap=inv_mycmap)
    ##plt.colorbar(cb)
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
    ax[0].set_title("Averaged dmdt for class4 ("+str(x_4.shape[0])+")"+"\nMin: "+str(round(four_avg.min(),5))+" Violet"+"\n Max: "+str(round(four_avg.max(),5))+" Yellow")
    ax[1].set_title("Averaged Change\n in Probabilities\n Min: "+str(round(m_4_avg.min(),5))+" Violet"+"\n Max: "+str(round(m_4_avg.max(),5))+" Yellow")
    fig.colorbar(im1,cax=cax1)
    fig.colorbar(im2,cax=cax2)
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("avg_sub/4_mean_heatmap.png")
    plt.close()
    
if len(five[0])!=0:
    fig,ax=plt.subplots(1,2)
    im1=ax[0].imshow(five_avg.reshape((23,24)))
    im2=ax[1].imshow(m_5_avg)
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    ##cb=ax.contourf(v,u,m_1_avg,15,cmap=inv_mycmap)
    ##plt.colorbar(cb)
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
    ax[0].set_title("Averaged dmdt for class5 ("+str(x_5.shape[0])+")"+"\nMin: "+str(round(five_avg.min(),5))+" Violet"+"\n Max: "+str(round(five_avg.max(),5))+" Yellow")
    ax[1].set_title("Averaged Change\n in Probabilities\n Min: "+str(round(m_5_avg.min(),5))+" Violet"+"\n Max: "+str(round(m_5_avg.max(),5))+" Yellow")
    fig.colorbar(im1,cax=cax1)
    fig.colorbar(im2,cax=cax2)
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("avg_sub/5_mean_heatmap.png")
    plt.close()

if len(six[0])!=0:
    fig,ax=plt.subplots(1,2)
    im1=ax[0].imshow(six_avg.reshape((23,24)))
    im2=ax[1].imshow(m_6_avg)
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    ##cb=ax.contourf(v,u,m_1_avg,15,cmap=inv_mycmap)
    ##plt.colorbar(cb)
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
    ax[0].set_title("Averaged dmdt for class6 ("+str(x_6.shape[0])+")"+"\nMin: "+str(round(six_avg.min(),5))+" Violet"+"\n Max: "+str(round(six_avg.max(),5))+" Yellow")
    ax[1].set_title("Averaged Change\n in Probabilities\n Min: "+str(round(m_6_avg.min(),5))+" Violet"+"\n Max: "+str(round(m_6_avg.max(),5))+" Yellow")
    fig.colorbar(im1,cax=cax1)
    fig.colorbar(im2,cax=cax2)
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("avg_sub/6_mean_heatmap.png")
    plt.close()

if len(eight[0])!=0:
    fig,ax=plt.subplots(1,2)
    im1=ax[0].imshow(eight_avg.reshape((23,24)))
    im2=ax[1].imshow(m_8_avg)
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    ##cb=ax.contourf(v,u,m_1_avg,15,cmap=inv_mycmap)
    ##plt.colorbar(cb)
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
    ax[0].set_title("Averaged dmdt for class8 ("+str(x_8.shape[0])+")"+"\nMin: "+str(round(eight_avg.min(),5))+" Violet"+"\n Max: "+str(round(eight_avg.max(),5))+" Yellow")
    ax[1].set_title("Averaged Change\n in Probabilities\n Min: "+str(round(m_8_avg.min(),5))+" Violet"+"\n Max: "+str(round(m_8_avg.max(),5))+" Yellow")
    fig.colorbar(im1,cax=cax1)
    fig.colorbar(im2,cax=cax2)
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("avg_sub/8_mean_heatmap.png")
    plt.close()
    
if len(thirteen[0])!=0:
    fig,ax=plt.subplots(1,2)
    im1=ax[0].imshow(thirteen_avg.reshape((23,24)))
    im2=ax[1].imshow(m_13_avg)
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    ##cb=ax.contourf(v,u,m_1_avg,15,cmap=inv_mycmap)
    ##plt.colorbar(cb)
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
    ax[0].set_title("Averaged dmdt for class13 ("+str(x_13.shape[0])+")"+"\nMin: "+str(round(thirteen_avg.min(),5))+" Violet"+"\n Max: "+str(round(thirteen_avg.max(),5))+" Yellow")
    ax[1].set_title("Averaged Change\n in Probabilities\n Min: "+str(round(m_13_avg.min(),5))+" Violet"+"\n Max: "+str(round(m_13_avg.max(),5))+" Yellow")
    fig.colorbar(im1,cax=cax1)
    fig.colorbar(im2,cax=cax2)
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("avg_sub/13_mean_heatmap.png")
    plt.close()










    


    
