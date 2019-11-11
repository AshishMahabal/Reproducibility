import matplotlib
matplotlib.use('Agg')

import generate

import numpy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from util import plot_confusion, plot_misclassifications

# parameters
seed = 0
verbose = 1
class_ = [1,2,4,5,6,8,13]
test_size = 0.2

classes =class_
# get data and encode labels
X_2d, X_features, y, indices = generate.get_data("all", classes=classes, shuffle=True, seed=seed)
labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(y)
y = labelencoder.transform(y).astype(numpy.int32)
print("Total number of instances: " + str(len(y)))

# split data (train/test)
X_train, X_test, y_train, y_test, indices_train, indices_test, X_2d_train, X_2d_test = train_test_split(X_features, y, indices, X_2d, test_size=test_size, random_state=seed)
print("Number of training instances: %i" % len(y_train))
print("Number of test instances: %i" % len(y_test))

# train forest
model = ExtraTreesClassifier(n_estimators=500, 
                             bootstrap=True, 
                             max_features=None, 
                             criterion="gini", 
                             n_jobs=-1, 
                             random_state=0)
model.fit(X_train, y_train)

# evaluate accuracy
preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)
acc = accuracy_score(y_test, preds)
print("Accuracy: %f" % acc)

y_test = labelencoder.inverse_transform(y_test)
preds = labelencoder.inverse_transform(preds)

# plot misclassifications
#plot_misclassifications(y_test, preds, X_2d_test, indices_test, "rf/misclassifications")

# save output
numpy.savetxt("rf/y_test_rf.csv", y_test, delimiter=",", fmt='%.4f')
numpy.savetxt("rf/preds_rf.csv", preds, delimiter=",", fmt='%.4f')
numpy.savetxt("rf/preds_proba_rf.csv", preds_proba, delimiter=",", fmt='%.4f')
plot_confusion(y_test, preds, "rf/confusion_rf.png")