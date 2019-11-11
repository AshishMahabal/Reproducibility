import numpy as np
import matplotlib.pyplot as plt
import os

shap_values = np.load("shap_values_2.5k.npy")
preds = np.load("preds.npy")
prob = np.load("preds_proba.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

os.mkdir("keras-vis/real/shap_sub/")
os.mkdir("keras-vis/bogus/shap_sub/")
#os.mkdir("shap/RRab/")
#os.mkdir("shap/RRc/")
#os.mkdir("shap/RRd/")
#os.mkdir("shap/RSCVn/")
#os.mkdir("shap/LPV/")

#index = {1:0, 2:1, 4:2, 5:3, 6:4, 8:5, 13:6}
index = {0:0, 1:1}
#clss = {1:'EW', 2:'EA', 4:'RRab', 5:'RRc', 6:'RRd', 8:'RSCVn', 13:'LPV'}
clss = {0:'bogus', 1:'real'}

for i in range(shap_values.shape[1]):
  #fig = plt.figure(figsize=(8, 2), dpi=100)
##  fig = plt.figure()
##  ax = fig.add_subplot(231)
##  ax.axis('off')
##  ax.imshow(x_test[i][:, :, 0], origin='upper', cmap=plt.cm.bone)
##  ax2 = fig.add_subplot(232)
##  ax2.axis('off')
##  ax2.imshow(x_test[i][:, :, 1], origin='upper', cmap=plt.cm.bone)
##  ax3 = fig.add_subplot(233)
##  ax3.axis('off')
##  ax3.imshow(x_test[i][:, :, 2], origin='upper', cmap=plt.cm.bone)
  
##  ax4 = fig.add_subplot(234)
##  ax4.axis('off')
##  ax4.imshow(shap_values[0][i][:,:,0], origin='upper')
##  ax5 = fig.add_subplot(235)
##  ax5.axis('off')
##  ax5.imshow(shap_values[0][i][:,:,1], origin='upper')
##  ax6 = fig.add_subplot(236)
  plt.figure()
  plt.axis('off')
  plt.imshow(shap_values[preds[i]][i][:,:,2])

  #plt.suptitle("SHAP viz. of  "+clss[preds[i]]+"\n"+"label:"+str(y_test[i])+
  #            "    Prediction:"+str(preds[i]))
  plt.savefig("keras-vis/"+clss[preds[i]]+"/shap_sub/"+str(i)+".png",
              bbox_inches='tight', transparent="True", pad_inches=0)
  plt.close()
