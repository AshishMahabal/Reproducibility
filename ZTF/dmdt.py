import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dmints = [-8,-5,-3,-2.5,-2,-1.5,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,5,8]
dtints = [0.0,1.0/145,2.0/145,3.0/145,4.0/145,1.0/25,2.0/25,3.0/25,1.5,2.5,3.5,4.5,5.5,7,10,20,30,60,90,120,240,600,960,2000,4000]
maxval = 255

l = glob.glob("3.1.lcs/*.lc")
lengthLC = []
lcid = []
x = []

for i in range(len(l)):
    df = pd.read_csv(l[i], delimiter=" ", header=None)
    if len(df) >= 2:
        lengthLC = lengthLC + [len(df)]
        lcid = lcid + [l[i].split("\\")[1]]
        #print(len(df))
        #print(l[i].split("\\")[1])
        mjd=pd.Series.as_matrix(df[0])
        mag=pd.Series.as_matrix(df[1])
        (smjd,smag) = list(zip(*sorted(list(zip(mjd,mag)))))
        #dmdt=np.zeros(shape=(len(dmints)-1,len(dtints)-1))
        maxpts = (len(mjd)*(len(mjd)-1))/2
        dmjd = []
        dmag = []

        # generate differences (w.r.t. time and mags)
        for i in range(len(mjd)):
          for j in range(i+1,len(mjd)):
              dmjd.append(mjd[j]-mjd[i])
              dmag.append(mag[j]-mag[i])

        # sort w.r.t. to first component (dmjd)
        (sdmjd,sdmag) = list(zip(*sorted(list(zip(dmjd,dmag)))))

        h=plt.hist2d(sdmjd,sdmag,bins=[dtints,dmints])
        dmdt=h[0]
        dmdt=np.transpose(dmdt)
        dmdt=maxval*dmdt/maxpts

        #np.save(name+"/indiv_dmdt/"+str(lcid[ii]),dmdt)

        x.append(dmdt)

np.save("lengthLC.npy", np.array(lengthLC))
np.save("lcid.npy", np.array(lcid))
np.save("X_ZTF.npy", np.array(x))


    

    
