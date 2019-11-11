import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import statistics as stats

# dictionary useful in indexing
# of loaded numpy arrays as each
# column in each of these numpy
# array represents a class

c={1:0,2:1,4:2,5:3,6:4,8:5,13:6} 

# creating a directory for storing the averaged results
os.mkdir("median")

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
##x_1=[]
##x_2=[]
##x_4=[]
##x_5=[]
##x_6=[]
##x_8=[]
##x_13=[]
##
### concatenating the objects belonging to a particular class
### into a list for each of the classes
##for i in range(len(one[0])):
##    x_1=x_1+[x[w[0][t[0][one[0][i]]]]]
##for i in range(len(two[0])):
##    x_2=x_2+[x[w[0][t[0][two[0][i]]]]]
##for i in range(len(four[0])):
##    x_4=x_4+[x[w[0][t[0][four[0][i]]]]]
##for i in range(len(five[0])):
##    x_5=x_5+[x[w[0][t[0][five[0][i]]]]]
##for i in range(len(six[0])):
##    x_6=x_6+[x[w[0][t[0][six[0][i]]]]]
##for i in range(len(eight[0])):
##    x_8=x_8+[x[w[0][t[0][eight[0][i]]]]]
##for i in range(len(thirteen[0])):
##    x_13=x_13+[x[w[0][t[0][thirteen[0][i]]]]]
##
### converting the concatenated lists into a numpy array
##x_1=np.array(x_1)
##x_2=np.array(x_2)
##x_4=np.array(x_4)
##x_5=np.array(x_5)
##x_6=np.array(x_6)
##x_8=np.array(x_8)
##x_13=np.array(x_13)

# loading the numpy arrays
x_1=np.load("avg/x_1.npy")
x_2=np.load("avg/x_2.npy")
x_4=np.load("avg/x_4.npy")
x_5=np.load("avg/x_5.npy")
x_6=np.load("avg/x_6.npy")
x_8=np.load("avg/x_8.npy")
x_13=np.load("avg/x_13.npy")

# computing average dmdt for each of the classes
if x_1.size!=0: # checking whether there exists a class which has no object being classified by cnn
    one_median=np.zeros((23,24))
    for j in range(x_1.shape[2]):
        for k in range(x_1.shape[3]):
            med=[]
            for i in range(x_1.shape[0]):
                med=med+[x_1[i,0,j,k]]
            one_median[j,k]=stats.median(med)
    np.save("median/1_median",one_median)
    
if x_2.size!=0:
    two_median=np.zeros((23,24))
    for j in range(x_2.shape[2]):
        for k in range(x_2.shape[3]):
            med=[]
            for i in range(x_2.shape[0]):
                med=med+[x_2[i,0,j,k]]
            two_median[j,k]=stats.median(med)
    np.save("median/2_median",two_median)

if x_4.size!=0:
    four_median=np.zeros((23,24))
    for j in range(x_4.shape[2]):
        for k in range(x_4.shape[3]):
            med=[]
            for i in range(x_4.shape[0]):
                med=med+[x_4[i,0,j,k]]
            four_median[j,k]=stats.median(med)
    np.save("median/4_median",four_median)

if x_5.size!=0:
    five_median=np.zeros((23,24))
    for j in range(x_5.shape[2]):
        for k in range(x_5.shape[3]):
            med=[]
            for i in range(x_5.shape[0]):
                med=med+[x_5[i,0,j,k]]
            five_median[j,k]=stats.median(med)
    np.save("median/5_median",five_median)

if x_6.size!=0:
    six_median=np.zeros((23,24))
    for j in range(x_6.shape[2]):
        for k in range(x_6.shape[3]):
            med=[]
            for i in range(x_6.shape[0]):
                med=med+[x_6[i,0,j,k]]
            six_median[j,k]=stats.median(med)
    np.save("median/6_median",six_median)

if x_8.size!=0:
    eight_median=np.zeros((23,24))
    for j in range(x_8.shape[2]):
        for k in range(x_8.shape[3]):
            med=[]
            for i in range(x_8.shape[0]):
                med=med+[x_8[i,0,j,k]]
            eight_median[j,k]=stats.median(med)
    np.save("median/8_median",eight_median)

if x_13.size!=0:
    thirteen_median=np.zeros((23,24))
    for j in range(x_13.shape[2]):
        for k in range(x_13.shape[3]):
            med=[]
            for i in range(x_13.shape[0]):
                med=med+[x_13[i,0,j,k]]
            thirteen_median[j,k]=stats.median(med)
    np.save("median/13_median",thirteen_median)

# loading m_a numpy array which contains information regarding
# the changes in probabilities encountered by blanking out individual pixels
m=np.load("m_a.npy")

# creating numpy arrays for getting the average change in probabilities for
# each of the classes
m_1=[]
m_2=[]
m_4=[]
m_5=[]
m_6=[]
m_8=[]
m_13=[]

# summing the changes in probabilities for each of the classes
for i in range(len(one[0])):
    m_1=m_1+[m[w[0][t[0][one[0][i]]]]]

for i in range(len(two[0])):
    m_2=m_2+[m[w[0][t[0][two[0][i]]]]]

for i in range(len(four[0])):
    m_4=m_4+[m[w[0][t[0][four[0][i]]]]]

for i in range(len(five[0])):
    m_5=m_5+[m[w[0][t[0][five[0][i]]]]]

for i in range(len(six[0])):
    m_6=m_6+[m[w[0][t[0][six[0][i]]]]]

for i in range(len(eight[0])):
    m_8=m_8+[m[w[0][t[0][eight[0][i]]]]]

for i in range(len(thirteen[0])):
    m_13=m_13+[m[w[0][t[0][thirteen[0][i]]]]]

m_1=np.array(m_1)
m_2=np.array(m_2)
m_4=np.array(m_4)
m_5=np.array(m_5)
m_6=np.array(m_6)
m_8=np.array(m_8)
m_13=np.array(m_13)

if m_1.size!=0: # checking whether there exists a class which has no object being classified by cnn
    m_1_median=np.zeros((23,24))
    for j in range(m_1.shape[1]):
        for k in range(m_1.shape[2]):
            med=[]
            for i in range(m_1.shape[0]):
                med=med+[m_1[i,j,k]]
            m_1_median[j,k]=stats.median(med)
    np.save("median/m_1_median",m_1_median)
    
if m_2.size!=0:
    m_2_median=np.zeros((23,24))
    for j in range(m_2.shape[1]):
        for k in range(m_2.shape[2]):
            med=[]
            for i in range(m_2.shape[0]):
                med=med+[m_2[i,j,k]]
            m_2_median[j,k]=stats.median(med)
    np.save("median/m_2_median",m_2_median)

if m_4.size!=0:
    m_4_median=np.zeros((23,24))
    for j in range(m_4.shape[1]):
        for k in range(m_4.shape[2]):
            med=[]
            for i in range(m_4.shape[0]):
                med=med+[m_4[i,j,k]]
            m_4_median[j,k]=stats.median(med)
    np.save("median/m_4_median",m_4_median)

if m_5.size!=0:
    m_5_median=np.zeros((23,24))
    for j in range(m_5.shape[1]):
        for k in range(m_5.shape[2]):
            med=[]
            for i in range(m_5.shape[0]):
                med=med+[m_5[i,j,k]]
            m_5_median[j,k]=stats.median(med)
    np.save("median/m_5_median",m_5_median)

if m_6.size!=0:
    m_6_median=np.zeros((23,24))
    for j in range(m_6.shape[1]):
        for k in range(m_6.shape[2]):
            med=[]
            for i in range(m_6.shape[0]):
                med=med+[m_6[i,j,k]]
            m_6_median[j,k]=stats.median(med)
    np.save("median/m_6_median",m_6_median)

if m_8.size!=0:
    m_8_median=np.zeros((23,24))
    for j in range(m_8.shape[1]):
        for k in range(m_8.shape[2]):
            med=[]
            for i in range(m_8.shape[0]):
                med=med+[m_8[i,j,k]]
            m_8_median[j,k]=stats.median(med)
    np.save("median/m_8_median",m_8_median)

if m_13.size!=0:
    m_13_median=np.zeros((23,24))
    for j in range(m_13.shape[1]):
        for k in range(m_13.shape[2]):
            med=[]
            for i in range(m_13.shape[0]):
                med=med+[m_13[i,j,k]]
            m_13_median[j,k]=stats.median(med)
    np.save("median/m_13_median",m_13_median)

### computing the average of changes in probabilities for each of the classes
##if len(one[0])!=0:# checking whether there exists a class which has no object being classified by cnn
##    m_1_median=m_1/len(one[0])
##    np.save("avg/m_1_mean",m_1_avg)
##if len(two[0])!=0:
##    m_2_avg=m_2/len(two[0])
##    np.save("avg/m_2_mean",m_2_avg)
##if len(four[0])!=0:
##    m_4_avg=m_4/len(four[0])
##    np.save("avg/m_4_mean",m_4_avg)
##if len(five[0])!=0:
##    m_5_avg=m_5/len(five[0])
##    np.save("avg/m_5_mean",m_5_avg)
##if len(six[0])!=0:
##    m_6_avg=m_6/len(six[0])
##    np.save("avg/m_6_mean",m_6_avg)
##if len(eight[0])!=0:
##    m_8_avg=m_8/len(eight[0])
##    np.save("avg/m_8_mean",m_8_avg)
##if len(thirteen[0])!=0:
##    m_13_avg=m_13/len(thirteen[0])
##    np.save("avg/m_13_mean",m_13_avg)

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
    fig,ax=plt.subplots(1,1)
    ax.imshow(one_median.reshape((23,24)))
    cb=ax.contourf(v,u,m_1_median,15,cmap=inv_mycmap)
    plt.colorbar(cb)
    plt.xticks(xloc,dtints,rotation=90)
    plt.yticks(yloc,dmints)
    plt.xlabel("dt(days)")#,labelpad=20)
    plt.ylabel("dm(mag)")
    plt.title("Image: Median dmdt for class1 ("+str(x_1.shape[0])+") \nHeatmap: Median Change in Probabilities")
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("median/1_median_heatmap.png")
    plt.close()

if len(two[0])!=0:
    fig,ax=plt.subplots(1,1)
    ax.imshow(two_median.reshape((23,24)))
    cb=ax.contourf(v,u,m_2_median,15,cmap=inv_mycmap)
    plt.colorbar(cb)
    plt.xticks(xloc,dtints,rotation=90)
    plt.yticks(yloc,dmints)
    plt.xlabel("dt(days)")#,labelpad=20)
    plt.ylabel("dm(mag)")
    plt.title("Image: Median dmdt for class2 ("+str(x_2.shape[0])+") \nHeatmap: Median Change in Probabilities")
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("median/2_median_heatmap.png")
    plt.close()

if len(four[0])!=0:
    fig,ax=plt.subplots(1,1)
    ax.imshow(four_median.reshape((23,24)))
    cb=ax.contourf(v,u,m_4_median,15,cmap=inv_mycmap)
    plt.colorbar(cb)
    plt.xticks(xloc,dtints,rotation=90)
    plt.yticks(yloc,dmints)
    plt.xlabel("dt(days)")#,labelpad=20)
    plt.ylabel("dm(mag)")
    plt.title("Image: Median dmdt for class4 ("+str(x_4.shape[0])+") \nHeatmap: Median Change in Probabilities")
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("median/4_median_heatmap.png")
    plt.close()
    
if len(five[0])!=0:
    fig,ax=plt.subplots(1,1)
    ax.imshow(five_median.reshape((23,24)))
    cb=ax.contourf(v,u,m_5_median,15,cmap=inv_mycmap)
    plt.colorbar(cb)
    plt.xticks(xloc,dtints,rotation=90)
    plt.yticks(yloc,dmints)
    plt.xlabel("dt(days)")#,labelpad=20)
    plt.ylabel("dm(mag)")
    plt.title("Image: Median dmdt for class5 ("+str(x_5.shape[0])+") \nHeatmap: Median Change in Probabilities")
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("median/5_median_heatmap.png")
    plt.close()

if len(six[0])!=0:
    fig,ax=plt.subplots(1,1)
    ax.imshow(six_median.reshape((23,24)))
    cb=ax.contourf(v,u,m_6_median,15,cmap=inv_mycmap)
    plt.colorbar(cb)
    plt.xticks(xloc,dtints,rotation=90)
    plt.yticks(yloc,dmints)
    plt.xlabel("dt(days)")#,labelpad=20)
    plt.ylabel("dm(mag)")
    plt.title("Image: Median dmdt for class6 ("+str(x_6.shape[0])+") \nHeatmap: Median Change in Probabilities")
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("median/6_median_heatmap.png")
    plt.close()

if len(eight[0])!=0:
    fig,ax=plt.subplots(1,1)
    ax.imshow(eight_median.reshape((23,24)))
    cb=ax.contourf(v,u,m_8_median,15,cmap=inv_mycmap)
    plt.colorbar(cb)
    plt.xticks(xloc,dtints,rotation=90)
    plt.yticks(yloc,dmints)
    plt.xlabel("dt(days)")#,labelpad=20)
    plt.ylabel("dm(mag)")
    plt.title("Image: Median dmdt for class8 ("+str(x_8.shape[0])+") \nHeatmap: Median Change in Probabilities")
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("median/8_median_heatmap.png")
    plt.close()
    
if len(thirteen[0])!=0:
    fig,ax=plt.subplots(1,1)
    ax.imshow(thirteen_median.reshape((23,24)))
    cb=ax.contourf(v,u,m_13_median,15,cmap=inv_mycmap)
    plt.colorbar(cb)
    plt.xticks(xloc,dtints,rotation=90)
    plt.yticks(yloc,dmints)
    plt.xlabel("dt(days)")#,labelpad=20)
    plt.ylabel("dm(mag)")
    plt.title("Image: Median dmdt for class13 ("+str(x_13.shape[0])+") \nHeatmap: Median Change in Probabilities")
    plt.tight_layout()
    #plt.ticks.set_xspacing(0.0005*mul)
    plt.savefig("median/13_median_heatmap.png")
    plt.close()










    


    
