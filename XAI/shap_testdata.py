import numpy as np
import shutil
import os

pred=np.load("preds.npy")
prob=np.load("preds_proba.npy")
y_test=np.load("y_test.npy")

classes=['EW','EA','RRab','RRc','RRd','RSCVn','LPV']
#classes=['RRab']
dp={'EW':1,'EA':2,'RRab':4,'RRc':5,'RRd':6,'RSCVn':8,'LPV':13}
ind={'EW':0,'EA':1,'RRab':2,'RRc':3,'RRd':4,'RSCVn':5,'LPV':6}
#os.mkdir("testdata/")
os.mkdir("testdata/shap/")

for clss in classes:
    os.mkdir("testdata/shap/"+clss+"/")
    clss_ind=np.where(pred==dp[clss])[0]
    for i in range(clss_ind.shape[0]):
        if prob[clss_ind[i],ind[clss]]>=0.98 and y_test[clss_ind[i]]==pred[clss_ind[i]]:
            shutil.copy2("trainingdata/shap/"+clss+"/"+str(clss_ind[i])+".png",
                         "testdata/shap/"+clss+"/"+str(clss_ind[i])+".png")
        

    
