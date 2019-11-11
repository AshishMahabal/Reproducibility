
# coding: utf-8

# In[48]:


import configparser
from six.moves import cPickle
import glob
import pandas as pd
import glob
import numpy as np
import os
import theano
import theano.gpuarray
theano.gpuarray.use("cuda" + str(0))
import matplotlib.pyplot as plt
from nolearn.lasagne import visualize
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
import glob
from texttable import Texttable

config = configparser.ConfigParser()
config.read('config_ztf.ini')


# In[8]:


# f=open(config['CNNs']['Periodic'],'rb')
# p_net=cPickle.load(f)
# f.close()

# f=open(config['CNNs']['Transient'],'rb')
# n_net=cPickle.load(f)
# f.close()

# f=open(config['CNNs']['Ensemble'],'rb')
# e_net=cPickle.load(f)
# f.close()
maxval=255.0

f=open(config['CNNs']['Model'],'rb')
net=cPickle.load(f)
f.close()

dmints=config['data']['dmints'].split(',')
dmints=[float(dmints[i]) for i in range(len(dmints))]

dtints=config['data']['dtints'].split(',')
dtints=[float(dtints[i]) for i in range(len(dtints))]

classes=config['data']['classes'].split(',')
classes=[int(classes[i]) for i in range(len(classes))]

l=glob.glob(config['data']['filename'])

header=config['csv']['header']

sep=config['csv']['delimiter'][1]

filt=config['csv']['filter']

lcid_present=config['csv']['lcid_present']

if lcid_present=='None':
  ss=config['csv']['lcid_start']
  ee=config['csv']['lcid_end']
  if ss!='None':
    ss=int(config['csv']['lcid_start'])  
  else:
    ss=None  
  if ee!='None':
    ee=int(config['csv']['lcid_end'])
  else:
    ee=None 
if header=='None':
  if lcid_present!='None':
    objid = int(config['csv']['lcid'])
  amjd = int(config['csv']['mjd'])
  amag = int(config['csv']['mag'])
  magerr = int(config['csv']['magerr'])
else:
  if lcid_present!='None':
    objid = config['csv']['lcid']
  amjd = config['csv']['mjd']
  amag = config['csv']['mag']
  magerr = config['csv']['magerr']

if filt!='None': 
  filter_column = config['csv']['filter_column']
  filter_type = config['csv']['filter_type']

x=[]
ids=[]
lengthLC=dict()

for q in range(len(l)):
    if q%10000==0:
      print(q)
    #name=l[q].split(".")[0].split("/")[1] #str(l[q])[:-4]
    #os.mkdir(name)
    #os.mkdir(name+"/indiv_dmdt/")
    if header=='None':
      df=pd.read_csv(l[q],header=None,delimiter=sep)
    else:
      df=pd.read_csv(l[q],delimiter=sep)
      
    if lcid_present=='None':
    
      name=l[q][ss-1:ee]
         
      if filt!='None':
        df2=df.loc[df[filter_column]==filter_type]
      else:
        df2=df
        
      lengthLC[name]=len(df2)
      
      if len(df2)>=2:
          #df3=df2[[amjd,amag]]
          #mjd=pd.Series.as_matrix(df3[amjd])
          #mag=pd.Series.as_matrix(df3[amag])

          #df3=df2[['objid','mjd','m']]
          #mjd=pd.Series.as_matrix(df3['mjd'])
          #mag=pd.Series.as_matrix(df3['m'])

          mjd=pd.Series.as_matrix(df2[amjd])
          mag=pd.Series.as_matrix(df2[amag])
          
          (smjd,smag) = list(zip(*sorted(list(zip(mjd,mag)))))
          #dmdt=np.zeros(shape=(len(dmints)-1,len(dtints)-1))
          maxpts = (len(mjd)*(len(mjd)-1))/2
          #dmjd = []
          #dmag = []

          # generate differences (w.r.t. time and mags)
##                for i in range(len(mjd)):
##                    for j in range(i+1,len(mjd)):
##                        dmjd.append(mjd[j]-mjd[i])
##                        dmag.append(mag[j]-mag[i])

          a_diff=np.array([((mjd[j]-mjd[i]),(mag[j]-mag[i])) for i in range(len(mjd)) for j in range(i+1,len(mjd))])
          dmjd=list(a_diff[:,0])
          dmag=list(a_diff[:,1])
                           
          # sort w.r.t. to first component (dmjd)
          (sdmjd,sdmag) = list(zip(*sorted(list(zip(dmjd,dmag)))))

          h=np.histogram2d(sdmjd,sdmag,bins=[dtints,dmints])
          dmdt=h[0]
          dmdt=np.transpose(dmdt)
          dmdt=(maxval*dmdt/maxpts+0.99999).astype(int)

          #np.save(name+"/indiv_dmdt/"+str(lcid[ii]),dmdt)
          
          x.append(dmdt)
          ids.append(name)

  

    else:
      
      lcid=sorted(set(df[objid]))
      
      for ii in range(len(lcid)):
          if filt=='None':
            df2=df[df[objid]==lcid[ii]]
            #df2=df1.loc[df[filter_column]==filter_type]
          else:
            df1=df[df[objid]==lcid[ii]]
            df2=df1.loc[df[filter_column]==filter_type]
          
          lengthLC[lcid[ii]]=len(df2)
          
          if len(df2)>=2:
              df3=df2[[amjd,amag]]
              mjd=pd.Series.as_matrix(df3[amjd])
              mag=pd.Series.as_matrix(df3[amag])
  
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
              dmdt=(maxval*dmdt/maxpts+0.99999).astype(int)
  
              #np.save(name+"/indiv_dmdt/"+str(lcid[ii]),dmdt)
              
              x.append(dmdt)
              ids.append(lcid[ii])
            
X_test=np.stack(x)
(ff,gg,hh)=X_test.shape
X=X_test.reshape(ff,1,gg,hh)
preds_prob=net.predict_proba(X)
preds=net.predict(X)
preds_prob=np.round(preds_prob, decimals=4)
ids=np.array(ids)
#dff=np.insert(preds_prob,0,ids,axis=1)
#print(dff)
#count=[len(np.where(preds==classes[i])[0]) for i in range(len(classes))]
#percent=[(count[i]/sum(count))*100.0 for i in range(len(classes))]

#df_res=pd.DataFrame(np.transpose(np.array([count,percent])), columns=['Nos','% of dataset'], index=classes)
#df_res=pd.DataFrame(dff)
df_res=pd.DataFrame(preds_prob)
#df_res.to_csv(config['output']['filename'], index=False, header=[""]+classes)
df_res.to_csv(config['output']['filename'], index=False, header=classes)
np.save(config['output']['dictionary']+".npy",lengthLC)
np.save(config['output']['ids']+".npy",ids)
            
            
            

            

