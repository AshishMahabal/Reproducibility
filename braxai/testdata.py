import numpy as np
import shutil
import os

pred=np.load("preds.npy")
prob=np.load("preds_proba.npy")
y_test=np.load("y_test.npy")

classes=['bogus','real']
#classes=['RRab']
dp={'bogus':0, 'real':1}
#ind={'EW':0,'EA':1,'RRab':2,'RRc':3,'RRd':4,'RSCVn':5,'LPV':6}
os.mkdir("testdata/")
os.mkdir("testdata/real/")
os.mkdir("testdata/bogus/")

for clss in classes:
    os.mkdir("testdata/"+clss+"/grad_CAM/")
    os.mkdir("testdata/"+clss+"/saliency/")
    os.mkdir("testdata/"+clss+"/blanking/")
    os.mkdir("testdata/"+clss+"/shap_sub/")
    clss_ind=np.where(pred==dp[clss])[0]
    for i in range(clss_ind.shape[0]):
        if prob[clss_ind[i],dp[clss]]>=0.98 and y_test[clss_ind[i]]==pred[clss_ind[i]]:
            shutil.copy2("trainingdata/"+clss+"/grad_CAM/"+str(clss_ind[i])+".png",
                         "testdata/"+clss+"/grad_CAM/"+str(clss_ind[i])+".png")
            shutil.copy2("trainingdata/"+clss+"/saliency/"+str(clss_ind[i])+".png",
                         "testdata/"+clss+"/saliency/"+str(clss_ind[i])+".png")
            shutil.copy2("trainingdata/"+clss+"/blanking/"+str(clss_ind[i])+".png",
                         "testdata/"+clss+"/blanking/"+str(clss_ind[i])+".png")
            shutil.copy2("trainingdata/"+clss+"/shap_sub/"+str(clss_ind[i])+".png",
                         "testdata/"+clss+"/shap_sub/"+str(clss_ind[i])+".png")
            
        

    
