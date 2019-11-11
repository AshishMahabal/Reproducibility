import numpy as np
import matplotlib.pyplot as plt
import os

shap_values = np.load("shap_values_5k.npy")
preds = np.load("preds.npy")
prob = np.load("preds_proba.npy")

os.mkdir("shap/")
os.mkdir("shap/EW/")
os.mkdir("shap/EA/")
os.mkdir("shap/RRab/")
os.mkdir("shap/RRc/")
os.mkdir("shap/RRd/")
os.mkdir("shap/RSCVn/")
os.mkdir("shap/LPV/")

index = {1:0, 2:1, 4:2, 5:3, 6:4, 8:5, 13:6}
clss = {1:'EW', 2:'EA', 4:'RRab', 5:'RRc', 6:'RRd', 8:'RSCVn', 13:'LPV'}

for i in range(shap_values.shape[1]):
  plt.figure()  
  plt.imshow(shap_values[index[preds[i]]][i][:,:,0])
  plt.axis('off')
  plt.savefig("shap/"+clss[preds[i]]+"/"+str(i)+".png",
              bbox_inches='tight', transparent="True", pad_inches=0)
  plt.close()
