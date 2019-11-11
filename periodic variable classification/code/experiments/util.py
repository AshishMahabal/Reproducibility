import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt

def _colorbar_fmt(x, pos):
    fm = '% *d' % (5, x)
    return fm

def ensure_dir(f):
    
    d = os.path.dirname(f)
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except:
            pass
        
def plot_confusion(y_test, preds, ofname):
    import seaborn
    from sklearn import metrics
    # confusion matrix:
    # y_true = [2, 0, 2, 2, 0, 1]
    # y_pred = [0, 0, 2, 2, 0, 2]
    # array([[2, 0, 0],
    #        [0, 0, 1],
    #        [1, 0, 2]])
    # x-axis: predictions, y-axis: true values (e.g., 2 prediction, but 1 as true label)
    cm = metrics.confusion_matrix(y_test, preds)
    c = cm
    cm = cm.astype('float')

    for i in range(cm.shape[0]):
        cm[i,:] *= 100/numpy.sum(cm[i,:])
    cm = cm.astype('int')


    classes = sorted(list(set(y_test)))

    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize = (10,10))
    seaborn.set(font_scale=3.0)
    seaborn.heatmap(df_cm, annot=True, fmt="d",  linewidths=.5,  cmap="YlGnBu", square=True, cbar=False)
    plt.xlabel("Prediction")
    plt.ylabel("True Class")
    plt.savefig(ofname, bbox_inches='tight')
    plt.close()
   
    df_c = pd.DataFrame(c, index=classes, columns=classes)
    plt.figure(figsize = (10,10))
    seaborn.set(font_scale=3.0)
    seaborn.heatmap(df_c, annot=True, fmt="d",  linewidths=.5,  cmap="YlGnBu", square=True, cbar=False)
    plt.xlabel("Prediction")
    plt.ylabel("True Class")
    plt.savefig("cnn/confusion_cnn.png", bbox_inches='tight')
    plt.close()


def plot_image(img, ofname, titles=None, figsize=(10,5)):
    """ Three sub-images given ...
    """
    
    fig = plt.figure(tight_layout=True, figsize=figsize)
    
    plt.imshow(img)
    plt.savefig(ofname, bbox_inches='tight')
    plt.close()
    
def plot_misclassifications(y, preds, X, orig_indices, odir, titles=None, verbose=0):
    
    orig_indices = numpy.array(orig_indices)
    
    misclassifications = numpy.array(range(len(y)))
    misclassifications = misclassifications[y != preds]
    misclassifications_indices = orig_indices[y != preds]
    
    if verbose > 0:
        print("Number of test elements: %i" % len(y))
        print("Misclassifications: %s" % str(misclassifications_indices))
        print("Plotting misclassifications ...")
        
    for i in range(len(misclassifications)):
        print("Plotting misclassification %i of %i ..." % (i, len(misclassifications)))
        index = misclassifications[i]
        orig_index = misclassifications_indices[i]
        
        ofname = os.path.join(odir, str(y[index]), str(orig_index) + "_" + str(preds[index]) + ".png")
        ensure_dir(ofname)
        plot_image(X[index], ofname, titles=titles)
         
